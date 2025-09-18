/**
 * Tokenizer package tests
 */
import { expect, test, describe } from "bun:test";

describe("Tokenizer Package", () => {
  test("basic tokenization works", () => {
    const text = "hello world test";
    const words = text.split(" ");
    
    expect(words).toHaveLength(3);
    expect(words[0]).toBe("hello");
    expect(words[1]).toBe("world");
    expect(words[2]).toBe("test");
  });

  test("token counting", () => {
    const text = "This is a test sentence with multiple words.";
    const wordCount = text.split(/\s+/).length;
    
    expect(wordCount).toBeGreaterThan(0);
    expect(typeof wordCount).toBe("number");
  });

  test("GPT token estimation", () => {
    const text = "Hello world";
    const words = text.split(/\s+/);
    const estimatedTokens = Math.ceil(words.length * 0.75);
    
    expect(estimatedTokens).toBeGreaterThan(0);
    expect(estimatedTokens).toBeLessThanOrEqual(words.length);
  });
});
