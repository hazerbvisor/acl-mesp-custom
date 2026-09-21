//
//  HazeCoderCodeTokenizer.swift
//  MLXMeSP
//
//  Phase 3: reversible code-first byte tokenizer + tiny authored code corpus.
//

import Foundation

public struct HazeCoderCodeTokenizer: Sendable {
    public let padToken: Int32 = 0
    public let bosToken: Int32 = 1
    public let eosToken: Int32 = 2
    public let fileSeparatorToken: Int32 = 3
    public let fimPrefixToken: Int32 = 4
    public let fimSuffixToken: Int32 = 5
    public let fimMiddleToken: Int32 = 6

    /// Byte IDs occupy 256...511. IDs 7...255 remain reserved for
    /// future learned/code-aware merges while the model vocabulary stays 4096.
    public let byteOffset: Int32 = 256

    public init() {}

    public var tokenizerVersion: String {
        "hazecoder-byte-v1"
    }

    public var decodableVocabularyLimit: Int {
        Int(byteOffset) + 256
    }

    public func encode(
        _ text: String,
        addBOS: Bool = false,
        addEOS: Bool = false
    ) -> [Int32] {
        var result = [Int32]()
        result.reserveCapacity(
            text.utf8.count + (addBOS ? 1 : 0) + (addEOS ? 1 : 0)
        )

        if addBOS {
            result.append(bosToken)
        }

        for byte in text.utf8 {
            result.append(byteOffset + Int32(byte))
        }

        if addEOS {
            result.append(eosToken)
        }

        return result
    }

    public func decode(
        _ tokenIDs: [Int32],
        stopAtEOS: Bool = true
    ) -> String {
        var bytes = [UInt8]()
        bytes.reserveCapacity(tokenIDs.count)

        for token in tokenIDs {
            if token == eosToken && stopAtEOS {
                break
            }

            if token >= byteOffset &&
               token < byteOffset + 256 {
                bytes.append(UInt8(token - byteOffset))
                continue
            }

            if token == fileSeparatorToken {
                bytes.append(contentsOf: [10, 10])
            }
        }

        return String(decoding: bytes, as: UTF8.self)
    }

    public func encodeFiles(_ files: [String]) -> [Int32] {
        var result = [Int32]()

        for (index, file) in files.enumerated() {
            result.append(bosToken)
            result.append(contentsOf: encode(file))
            result.append(eosToken)

            if index + 1 < files.count {
                result.append(fileSeparatorToken)
            }
        }

        return result
    }
}

public enum HazeCoderTinyCodeCorpus {
    /// Small, original examples authored for the first real-text training proof.
    /// This is intentionally tiny: the purpose is to validate the tokenizer,
    /// real-code batching, checkpointing and generation pipeline on-device.
    public static let samples: [String] = [
        """
        def add(a, b):
            return a + b

        def multiply(a, b):
            return a * b

        print(add(2, 3))
        """,
        """
        def clamp(value, low, high):
            if value < low:
                return low
            if value > high:
                return high
            return value
        """,
        """
        func add(_ a: Int, _ b: Int) -> Int {
            return a + b
        }

        let result = add(2, 3)
        print(result)
        """,
        """
        func isEven(_ value: Int) -> Bool {
            return value % 2 == 0
        }

        for value in 0..<8 {
            print(value, isEven(value))
        }
        """,
        """
        #include <stdio.h>

        int add(int a, int b) {
            return a + b;
        }

        int main(void) {
            printf("%d\\n", add(2, 3));
            return 0;
        }
        """,
        """
        fn add(a: i32, b: i32) -> i32 {
            a + b
        }

        fn main() {
            println!("{}", add(2, 3));
        }
        """,
        """
        const add = (a, b) => {
            return a + b;
        };

        console.log(add(2, 3));
        """,
        """
        struct Counter {
            var value: Int = 0

            mutating func increment() {
                value += 1
            }
        }
        """
    ]

    public static let defaultPrompt = "func add(_ a: Int, _ b: Int) -> Int {\n    "
}
