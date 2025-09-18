/**
 * Basic functionality tests
 */
import { expect, test, describe } from "bun:test";

describe("Basic Test Suite", () => {
  test("basic functionality works", () => {
    expect(1 + 1).toBe(2);
    expect("hello").toBe("hello");
  });

  test("array operations", () => {
    const arr = [1, 2, 3];
    expect(arr).toHaveLength(3);
    expect(arr[0]).toBe(1);
  });
});
