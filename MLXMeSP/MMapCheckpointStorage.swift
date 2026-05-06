//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  MMapCheckpointStorage.swift
//  mlx-mesp
//
//  Created by Congzheng Song on 9/18/25.
//

import Foundation
import MLX
import Cmlx

extension DType {
    init(_ cmlxDtype: mlx_dtype) throws {
        switch cmlxDtype {
        case MLX_BOOL: self = .bool
        case MLX_UINT8: self = .uint8
        case MLX_UINT16: self = .uint16
        case MLX_UINT32: self = .uint32
        case MLX_UINT64: self = .uint64
        case MLX_INT8: self = .int8
        case MLX_INT16: self = .int16
        case MLX_INT32: self = .int32
        case MLX_INT64: self = .int64
        case MLX_FLOAT16: self = .float16
        case MLX_FLOAT32: self = .float32
        case MLX_BFLOAT16: self = .bfloat16
        case MLX_COMPLEX64: self = .complex64
        case MLX_FLOAT64: self = .float64
        default:
            throw MLXError.caught("Unsupported dtype: \(cmlxDtype)")
        }
    }
}

/// A true memory-mapped storage for MLXArray checkpoints using mmap syscalls
/// This provides zero-copy access to checkpoint data with OS-managed virtual memory
public class MMapCheckpointStorage {
    private var storage: [String: MMapEntry] = [:]
    private let baseDirectory: URL
    private let fileManager = FileManager.default
    
    public struct MMapEntry {
        let fileURL: URL
        let shape: [Int]
        let dtype: DType
        let fileDescriptor: Int32
        let mappedPointer: UnsafeMutableRawPointer?
        let mappedSize: Int
        
        var isMapped: Bool { mappedPointer != nil }
        
        init(fileURL: URL, shape: [Int], dtype: DType, fileDescriptor: Int32, mappedPointer: UnsafeMutableRawPointer?, mappedSize: Int) {
            self.fileURL = fileURL
            self.shape = shape
            self.dtype = dtype
            self.fileDescriptor = fileDescriptor
            self.mappedPointer = mappedPointer
            self.mappedSize = mappedSize
        }
    }
    
    /// Header structure for mmap files
    private struct MMapHeader {
        let magic: UInt32 = 0x4D4C5841 // "MLXA"
        let version: UInt32 = 1
        let dtype: UInt32
        let ndim: UInt32
        let shapes: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) // Max 8 dimensions
        let dataOffset: UInt64
        let dataSize: UInt64
        
        static let size = MemoryLayout<MMapHeader>.size
        
        init(dtype: DType, shape: [Int], dataSize: Int) {
            self.dtype = dtype.cmlxDtype.rawValue
            self.ndim = UInt32(shape.count)
            
            var shapesTuple = (Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0))
            for (i, dim) in shape.enumerated() {
                guard i < 8 else { break }
                switch i {
                case 0: shapesTuple.0 = Int32(dim)
                case 1: shapesTuple.1 = Int32(dim)
                case 2: shapesTuple.2 = Int32(dim)
                case 3: shapesTuple.3 = Int32(dim)
                case 4: shapesTuple.4 = Int32(dim)
                case 5: shapesTuple.5 = Int32(dim)
                case 6: shapesTuple.6 = Int32(dim)
                case 7: shapesTuple.7 = Int32(dim)
                default: break
                }
            }
            self.shapes = shapesTuple
            
            self.dataOffset = UInt64(MMapHeader.size)
            self.dataSize = UInt64(dataSize)
        }
        
        func getShape() -> [Int] {
            let shapesArray = [shapes.0, shapes.1, shapes.2, shapes.3, shapes.4, shapes.5, shapes.6, shapes.7]
            return Array(shapesArray.prefix(Int(ndim)).map(Int.init))
        }
        
        func getDType() throws -> DType {
            // C enums don't have failable initializers - create directly
            let cmlxDtype = mlx_dtype(rawValue: dtype)
            return try DType(cmlxDtype)
        }
    }
    
    public init(baseDirectory: URL? = nil) throws {
        if let baseDirectory = baseDirectory {
            self.baseDirectory = baseDirectory
        } else {
            // Create temporary directory for mmap files
            self.baseDirectory = fileManager.temporaryDirectory.appendingPathComponent("MMapCheckpointStorage\(UUID().uuidString)")
        }
        
        try fileManager.createDirectory(at: self.baseDirectory, withIntermediateDirectories: true)
    }
    
    deinit {
        cleanup()
        try? fileManager.removeItem(at: baseDirectory)
    }
    
    /// Dictionary-like subscript access
    public subscript(key: String) -> MLXArray? {
        get {
            return getArray(for: key)
        }
        set {
            if let newValue = newValue {
                setArray(newValue, for: key)
            } else {
                removeArray(for: key)
            }
        }
    }
    
    /// Store an MLXArray using true memory mapping
    public func setArray(_ array: MLXArray, for key: String) {
        // Ensure array is evaluated and on CPU
        array.eval()

        let fileURL = baseDirectory.appendingPathComponent("\(key).mmap")
        let filePath = fileURL.path

        do {
            // Get array data
            let arrayData = array.asData().data
            let dataSize = arrayData.count
            let totalSize = MMapHeader.size + dataSize
            
            // Create and open file
            let fd = open(filePath, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR)
            guard fd >= 0 else {
                throw MLXError.caught("Failed to create file for key '\(key)': \(String(cString: strerror(errno)))")
            }
            
            // Resize file to required size
            guard ftruncate(fd, off_t(totalSize)) == 0 else {
                close(fd)
                throw MLXError.caught("Failed to resize file for key '\(key)': \(String(cString: strerror(errno)))")
            }
            
            // Memory map the file
            let mappedPtr = mmap(nil, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
            guard let mappedPtr, mappedPtr != MAP_FAILED else {
                close(fd)
                throw MLXError.caught("Failed to mmap file for key '\(key)': \(String(cString: strerror(errno)))")
            }
            
            let typedPtr = mappedPtr.assumingMemoryBound(to: UInt8.self)
            
            // Write header
            let header = MMapHeader(dtype: array.dtype, shape: array.shape, dataSize: dataSize)
            withUnsafeBytes(of: header) { headerBytes in
                typedPtr.update(from: headerBytes.bindMemory(to: UInt8.self).baseAddress!, count: MMapHeader.size)
            }
            
            // Write array data
            arrayData.withUnsafeBytes { dataBytes in
                (typedPtr + MMapHeader.size).update(from: dataBytes.bindMemory(to: UInt8.self).baseAddress!, count: dataSize)
            }
            
            // Sync to disk
            msync(mappedPtr, totalSize, MS_ASYNC)
            
            // Store entry
            let entry = MMapEntry(
                fileURL: fileURL,
                shape: array.shape,
                dtype: array.dtype,
                fileDescriptor: fd,
                mappedPointer: mappedPtr,
                mappedSize: totalSize
            )
            
            // Clean up any existing entry
            if let existingEntry = storage[key] {
                cleanupEntry(existingEntry)
            }
            
            storage[key] = entry
            
        } catch {
            fatalError("Failed to store array for key '\(key)': \(error)")
        }
    }
    
    /// Retrieve an MLXArray using memory mapping
    public func getArray(for key: String) -> MLXArray? {
        guard let entry = storage[key] else { return nil }

        guard entry.isMapped, let mappedPtr = entry.mappedPointer else {
            // Try to remap if not currently mapped
            return remapAndGetArray(for: key)
        }

        do {
            // Read header
            let headerPtr = mappedPtr.assumingMemoryBound(to: MMapHeader.self)
            let header = headerPtr.pointee

            // Validate header
            guard header.magic == 0x4D4C5841, header.version == 1 else {
                throw MLXError.caught("Invalid mmap file header for key '\(key)'")
            }

            // Get data pointer
            let dataPtr = mappedPtr.advanced(by: Int(header.dataOffset))
            let dataSize = Int(header.dataSize)

            // Create data from mapped memory (zero-copy)
            let data = Data(bytesNoCopy: dataPtr, count: dataSize, deallocator: .none)

            // Reconstruct MLXArray from data
            let shape = header.getShape()
            let dtype = try header.getDType()

            let result = MLXArray(data, shape, dtype: dtype)
            return result
            
        } catch {
            fatalError("Failed to load array for key '\(key)': \(error)")
        }
    }
    
    /// Remap a file that's not currently mapped
    private func remapAndGetArray(for key: String) -> MLXArray? {
        guard let entry = storage[key] else { return nil }
        
        let filePath = entry.fileURL.path
        
        do {
            // Open file if not already open
            var fd = entry.fileDescriptor
            if fd < 0 {
                fd = open(filePath, O_RDWR)
                guard fd >= 0 else {
                    throw MLXError.caught("Failed to reopen file for key '\(key)'")
                }
            }
            
            // Get file size
            var stat = stat()
            guard fstat(fd, &stat) == 0 else {
                if entry.fileDescriptor < 0 { close(fd) }
                throw MLXError.caught("Failed to get file size for key '\(key)'")
            }
            
            let fileSize = Int(stat.st_size)
            
            // Memory map the file
            let mappedPtr = mmap(nil, fileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
            guard mappedPtr != MAP_FAILED else {
                if entry.fileDescriptor < 0 { close(fd) }
                throw MLXError.caught("Failed to remap file for key '\(key)'")
            }
            
            // Update entry
            let newEntry = MMapEntry(
                fileURL: entry.fileURL,
                shape: entry.shape,
                dtype: entry.dtype,
                fileDescriptor: fd,
                mappedPointer: mappedPtr,
                mappedSize: fileSize
            )
            
            storage[key] = newEntry
            
            // Now get the array
            return getArray(for: key)
            
        } catch {
            fatalError("Failed to remap and load array for key '\(key)': \(error)")
        }
    }
    
    /// Remove an array from storage
    public func removeArray(for key: String) {
        guard let entry = storage[key] else { return }
        
        cleanupEntry(entry)
        storage.removeValue(forKey: key)
        
        // Remove file
        try? fileManager.removeItem(at: entry.fileURL)
    }
    
    /// Clean up a single entry's resources
    private func cleanupEntry(_ entry: MMapEntry) {
        if let mappedPtr = entry.mappedPointer {
            munmap(mappedPtr, entry.mappedSize)
        }
        if entry.fileDescriptor >= 0 {
            close(entry.fileDescriptor)
        }
    }
    
    public func merge(_ other: [String: MLXArray], uniquingKeysWith combine: (MLXArray, MLXArray) throws -> MLXArray = { _, new in new }) rethrows {
        for (key, array) in other {
            if let existing = self[key] {
                self[key] = try combine(existing, array)
            } else {
                self[key] = array
            }
        }
    }
    
    /// Get all keys currently stored
    public var keys: Dictionary<String, MMapEntry>.Keys {
        return storage.keys
    }
    
    /// Get count of stored arrays
    public var count: Int {
        return storage.count
    }
    
    /// Check if storage is empty
    public var isEmpty: Bool {
        return storage.isEmpty
    }
    
    /// Force cleanup of all files and memory mappings
    public func cleanup() {
        for entry in storage.values {
            cleanupEntry(entry)
        }
        storage.removeAll()
    }
}
