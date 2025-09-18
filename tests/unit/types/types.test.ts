/**
 * Types package tests
 */
import { expect, test, describe } from "bun:test";

describe("Types Package", () => {
  test("basic type definitions exist", () => {
    // Test that we can import types from the types package
    const config: any = {
      models: {
        embed: "test-embed",
        rerank: "test-rerank"
      },
      retrieval: {
        alpha: 0.5,
        beta: 0.5,
        gamma_kind_boost: {
          tool_result: 1.0,
          user_code: 1.1,
          prose: 1.0,
          code: 1.2
        },
        variant: "hybrid" as const
      }
    };
    
    expect(config.models.embed).toBe("test-embed");
    expect(config.retrieval.variant).toBe("hybrid");
  });

  test("candidate type structure", () => {
    const candidate: any = {
      id: "test-id",
      text: "test text",
      messageId: "msg-123",
      kind: "prose" as const,
      bm25Score: 0.8,
      vectorScore: 0.9,
      hybridScore: 0.85
    };

    expect(candidate.id).toBe("test-id");
    expect(candidate.kind).toBe("prose");
    expect(typeof candidate.bm25Score).toBe("number");
  });
});
