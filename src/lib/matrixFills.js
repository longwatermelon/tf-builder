import { identity, zeros } from "./linalg";

// quick fills offered under weight matrices in the editor
export const FILL_ZERO = { label: "0", title: "fill with zeros", build: (rows, cols) => zeros(rows, cols) };

export const FILL_IDENTITY = { label: "I", title: "fill with the identity", build: (rows, cols) => identity(rows, cols) };

export const FILL_CAUSAL = {
  label: "causal",
  title: "0 at j ≤ i, -inf above the diagonal",
  build: (rows, cols) =>
    Array.from({ length: rows }, (_, i) => Array.from({ length: cols }, (_, j) => (j <= i ? 0 : -Infinity))),
};
