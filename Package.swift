// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MetalKernels",
    platforms: [
        .macOS(.v12)
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "MetalKernels",
            dependencies: []
        )
    ]
)
