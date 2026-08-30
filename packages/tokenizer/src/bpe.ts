/**
 * The classic BPE merge loop (same algorithm GPT-2's original code and
 * every HF "slow" BPE tokenizer use): repeatedly merge the pair of adjacent
 * symbols with the lowest merge rank, until no known pair remains.
 * Architecture-agnostic — GPT-2 and Llama differ only in how they produce
 * the *initial* symbol list fed into this function (see gpt2Pretokenize.ts
 * / llamaPretokenize.ts), not in the merge algorithm itself.
 */
export function bpeMerge(symbols: string[], mergeRank: Map<string, number>): string[] {
  if (symbols.length <= 1) return symbols;
  let word = symbols;

  for (;;) {
    let bestRank = Infinity;
    let bestIndex = -1;
    for (let i = 0; i < word.length - 1; i++) {
      const rank = mergeRank.get(word[i] + " " + word[i + 1]);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIndex = i;
      }
    }
    if (bestIndex === -1) break;

    const merged = word[bestIndex] + word[bestIndex + 1];
    const next: string[] = word.slice(0, bestIndex);
    next.push(merged);
    let i = bestIndex + 2;
    while (i < word.length) {
      // also merge any further non-overlapping occurrences of the same pair in one pass
      if (i < word.length - 1 && word[i] === word[bestIndex] && word[i + 1] === word[bestIndex + 1]) {
        next.push(merged);
        i += 2;
      } else {
        next.push(word[i]);
        i += 1;
      }
    }
    word = next;
    if (word.length === 1) break;
  }

  return word;
}
