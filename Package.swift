// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
//  For licensing see accompanying LICENSE file.
//  Copyright (C) 2025 Apple Inc. All Rights Reserved.
//

import PackageDescription

let package = Package(
    name: "mlx-mesp",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(
            name: "MLXMeSP",
            targets: ["MLXMeSP"]),
    ],
    dependencies: [
        .package(url: "https://github.com/congzheng-song/mlx-swift", branch: "mebp"),
        .package(
            url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "0.1.23")
        ),
    ],
    targets: [
        .target(
            name: "MLXMeSP",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "MLXMeSP",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "MLXMeSPTests",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
                "MLXMeSP",
            ],
            path: "MLXMeSPTests",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)

