import Foundation

/// XTool Mobile bootstrap compatibility surface for swift-transformers 0.1.24.
///
/// The first MeSP milestone uses pre-tokenized WikiText input and does not render
/// chat templates. Keeping this tiny module lets the upstream Tokenizers sources
/// compile unchanged without vendoring the full Jinja + OrderedCollections stack.
///
/// Replace this target with the real Jinja package before enabling chat-format
/// training through applyChatTemplate(...).
public enum JinjaUnavailableError: LocalizedError, Sendable {
    case chatTemplatesNotEnabled

    public var errorDescription: String? {
        "Jinja chat-template rendering is not enabled in the XTool Mobile pre-tokenized bootstrap."
    }
}

public final class Template: @unchecked Sendable {
    private let source: String

    public init(_ source: String) throws {
        self.source = source
    }

    public func render(_ context: [String: Any]) throws -> String {
        _ = source
        _ = context
        throw JinjaUnavailableError.chatTemplatesNotEnabled
    }
}
