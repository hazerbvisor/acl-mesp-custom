//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//
//  MMapCheckpointStorageTests.swift
//  mlx-mesp
//
//  Created by Congzheng Song on 9/18/25.
//

import XCTest
import MLX
@testable import MLXMeSP

final class MMapCheckpointStorageTests: XCTestCase {
    
    var tempDirectory: URL!
    var storage: MMapCheckpointStorage!
    
    override func setUp() {
        super.setUp()
        tempDirectory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try! FileManager.default.createDirectory(at: tempDirectory, withIntermediateDirectories: true)
        storage = try! MMapCheckpointStorage(baseDirectory: tempDirectory)
    }
    
    override func tearDown() {
        storage?.cleanup()
        try? FileManager.default.removeItem(at: tempDirectory)
        super.tearDown()
    }
    
    func allClose(_ x: MLXArray, _ y: MLXArray, rtol: Double = 1e-8, atol: Double = 1e-8) -> Bool {
        MLX.allClose(x, y, rtol: rtol, atol: atol).item(Bool.self)
    }
    
    func testBasicSetAndGet() throws {
        let testArray = MLXArray([1, 2, 3, 4, 5])
        storage.setArray(testArray, for: "test_array")
        let retrievedArray = storage.getArray(for: "test_array")!
        XCTAssertTrue(allClose(testArray, retrievedArray))
    }
    
    func testMultipleArrays() throws {
        let arrays = [
            "array1": MLXArray([1.0, 2.0, 3.0] as [Float32]),
            "array2": MLXArray([1, 2, 3, 4], [2, 2]),
            "array3": MLXArray(Float32(42.0))
        ]
        
        for (key, array) in arrays {
            storage.setArray(array, for: key)
        }
        
        for (key, expectedArray) in arrays {
            let retrievedArray = storage.getArray(for: key)!
            XCTAssertTrue(allClose(expectedArray, retrievedArray))
        }
    }
    
    func testDifferentDataTypes() throws {
        let testCases: [(String, MLXArray)] = [
            ("float32", MLXArray([1.1, 2.2, 3.3] as [Float32])),
            ("int32", MLXArray([10, 20, 30] as [Int32])),
            ("bool", MLXArray([true, false, true])),
            ("uint8", MLXArray([255, 128, 0] as [UInt8]))
        ]
        
        for (key, array) in testCases {
            storage.setArray(array, for: key)
        }
        
        for (key, expectedArray) in testCases {
            let retrievedArray = storage.getArray(for: key)!
            XCTAssertEqual(expectedArray.dtype, retrievedArray.dtype, "Data types should match for \(key)")
            XCTAssertTrue(allClose(expectedArray, retrievedArray, rtol: 1e-5), "Arrays should be equal for \(key)")
        }
    }
    
    func testLargeArrays() throws {
        let largeArray = MLXRandom.normal([1000, 1000])
        
        storage.setArray(largeArray, for: "large_array")
        let retrievedArray = storage.getArray(for: "large_array")!
        
        XCTAssertEqual(largeArray.shape, retrievedArray.shape)
        XCTAssertEqual(largeArray.dtype, retrievedArray.dtype)
        XCTAssertTrue(allClose(largeArray, retrievedArray))
    }
    
    func testArrayOverwrite() throws {
        let originalArray = MLXArray([1, 2, 3])
        let newArray = MLXArray([4, 5, 6])
        
        storage.setArray(originalArray, for: "test_key")
        let retrievedOriginal = storage.getArray(for: "test_key")!
        XCTAssertTrue(allClose(originalArray, retrievedOriginal))
        
        storage.setArray(newArray, for: "test_key")
        let retrievedNew = storage.getArray(for: "test_key")!
        XCTAssertTrue(allClose(newArray, retrievedNew, rtol: 1e-5))
    }
    
    func testNonexistentKey() throws {
        XCTAssertNil(storage.getArray(for: "nonexistent_key"))
    }

    func testRemoveArray() throws {
        let testArray = MLXArray([1, 2, 3])

        storage.setArray(testArray, for: "test_key")
        XCTAssertTrue(storage.keys.contains("test_key"))

        storage.removeArray(for: "test_key")
        XCTAssertFalse(storage.keys.contains("test_key"))

        XCTAssertNil(storage.getArray(for: "test_key"))
    }
}
