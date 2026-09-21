#if XTOOL_MOBILE
import Cmlx
import Foundation
import MLX

/// XTool-only replacement for the MeBP fork's
/// ImportedFunction.call(args:kwargs:) convenience API.
///
/// Official MLX exposes ImportedFunction as @dynamicCallable, which cannot accept
/// a runtime [MLXArray] directly. MeSP's exported run configs assemble a dynamic
/// argument list, so reproduce the fork's behavior against the public Cmlx API.
final class XToolImportedFunction {
    private let ctx: mlx_imported_function

    init(url: URL) throws {
        self.ctx = try withError {
            mlx_imported_function_new(url.path)
        }
    }

    deinit {
        mlx_imported_function_free(ctx)
    }

    func call(args: [MLXArray], kwargs: [String: MLXArray]) throws -> [MLXArray] {
        var result = mlx_vector_array_new()
        defer { mlx_vector_array_free(result) }

        let positionalArgs = mlx_vector_array_new()
        defer { mlx_vector_array_free(positionalArgs) }
        for value in args {
            mlx_vector_array_append_value(positionalArgs, value.ctx)
        }

        let keywordArgs = mlx_map_string_to_array_new()
        defer { mlx_map_string_to_array_free(keywordArgs) }
        for (key, value) in kwargs {
            key.withCString { cKey in
                mlx_map_string_to_array_insert(keywordArgs, cKey, value.ctx)
            }
        }

        _ = try withError {
            mlx_imported_function_apply_kwargs(&result, ctx, positionalArgs, keywordArgs)
        }

        return (0 ..< mlx_vector_array_size(result)).map { index in
            var array = mlx_array_new()
            mlx_vector_array_get(&array, result, index)
            return MLXArray(array)
        }
    }
}
#endif
