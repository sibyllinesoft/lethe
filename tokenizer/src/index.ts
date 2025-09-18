/**
 * GPT tokenizer utilities for Lethe
 */
import natural from 'natural';

export class BasicTokenizer {
  private tokenizer: any;

  constructor() {
    this.tokenizer = new natural.WordTokenizer();
  }

  /**
   * Tokenize text into an array of tokens
   */
  tokenize(text: string): string[] {
    return this.tokenizer.tokenize(text.toLowerCase()) || [];
  }

  /**
   * Count tokens in text
   */
  countTokens(text: string): number {
    return this.tokenize(text).length;
  }

  /**
   * Estimate GPT-4 token count (rough approximation)
   * GPT-4 uses ~0.75 tokens per word on average
   */
  estimateGPTTokens(text: string): number {
    const wordCount = this.tokenize(text).length;
    return Math.ceil(wordCount * 0.75);
  }
}

export const tokenizer = new BasicTokenizer();

export default tokenizer;
