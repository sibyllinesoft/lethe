use serde::{Deserialize, Serialize};
use std::f64::consts::LN_2;
use xxhash_rust::xxh3::xxh3_64_with_seed;

/// Simple Bloom filter implementation tailored for text token checks
#[derive(Clone, Serialize, Deserialize)]
pub struct SimpleBloom {
    num_bits: usize,
    num_hashes: u32,
    bits: Vec<u8>,
}

impl std::fmt::Debug for SimpleBloom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimpleBloom")
            .field("num_bits", &self.num_bits)
            .field("num_hashes", &self.num_hashes)
            .finish()
    }
}

impl SimpleBloom {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let expected_items = expected_items.max(1);
        let false_positive_rate = false_positive_rate.clamp(1e-6, 0.5);

        let m = ((-(expected_items as f64) * false_positive_rate.ln()) / (LN_2.powi(2))).ceil()
            as usize;
        let num_bits = m.max(64);
        let num_hashes = ((num_bits as f64 / expected_items as f64) * LN_2)
            .round()
            .max(1.0) as u32;

        let byte_len = (num_bits + 7) / 8;

        Self {
            num_bits,
            num_hashes,
            bits: vec![0; byte_len],
        }
    }

    #[inline]
    fn set_bit(&mut self, idx: usize) {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        self.bits[byte_idx] |= 1 << bit_idx;
    }

    #[inline]
    fn test_bit(&self, idx: usize) -> bool {
        let byte_idx = idx / 8;
        let bit_idx = idx % 8;
        (self.bits[byte_idx] & (1 << bit_idx)) != 0
    }

    #[inline]
    fn hash(&self, item: &str, seed: u64) -> usize {
        (xxh3_64_with_seed(item.as_bytes(), seed) as usize) % self.num_bits
    }

    pub fn insert(&mut self, item: &str) {
        let item = item.trim();
        if item.is_empty() {
            return;
        }
        let h1 = self.hash(item, 0);
        let h2 = self.hash(item, 1);
        for i in 0..self.num_hashes {
            let combined = (h1 + i as usize * h2) % self.num_bits;
            self.set_bit(combined);
        }
    }

    pub fn contains(&self, item: &str) -> bool {
        let item = item.trim();
        if item.is_empty() {
            return true;
        }
        let h1 = self.hash(item, 0);
        let h2 = self.hash(item, 1);
        for i in 0..self.num_hashes {
            let combined = (h1 + i as usize * h2) % self.num_bits;
            if !self.test_bit(combined) {
                return false;
            }
        }
        true
    }

    pub fn contains_any<'a, I>(&self, items: I) -> bool
    where
        I: IntoIterator<Item = &'a str>,
    {
        items.into_iter().any(|item| self.contains(item))
    }
}
