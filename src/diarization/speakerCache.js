// Arrival-Order Speaker Cache (AOSC) + FIFO。バッチサイズ 1 前提。
// transformers の Nemotron3DiarizationSpeakerCache (Apache-2.0) を JS に移植したもの。
// 埋め込みは Float32Array(512)、確率は Float32Array(8) の配列で持つ。

const LOG_HALF = Math.log(0.5);

// config.json の streaming_config の値
const POLICY = {
  cacheLength: 264,
  silenceFramesPerSpeaker: 1,
  predictionScoreThreshold: 0.25,
  latestFramesScoreBoost: 0.05,
  strongBoostRate: 0.75,
  weakBoostRate: 1.5,
  minPositiveScoresRate: 0.5,
};

export class SpeakerCache {
  /**
   * @param {number} numSpeakers
   * @param {number} fifoLength FIFO の容量 (encoder フレーム)
   * @param {number} updatePeriod FIFO があふれたときにキャッシュへ移す最小フレーム数
   * @param {Float32Array} silenceEmbed 学習済みの無音埋め込み
   */
  constructor(numSpeakers, fifoLength, updatePeriod, silenceEmbed) {
    this.numSpeakers = numSpeakers;
    this.fifoLength = fifoLength;
    this.updatePeriod = updatePeriod;
    this.silenceEmbed = silenceEmbed;
    this.cacheEmbeds = [];
    this.cacheProbs = [];
    this.fifo = [];
    this.isCompressed = false;

    const budget = Math.floor(POLICY.cacheLength / numSpeakers) - POLICY.silenceFramesPerSpeaker;
    this.minPositiveScores = Math.floor(budget * POLICY.minPositiveScoresRate);
    this.numStrongBoosted = Math.floor(budget * POLICY.strongBoostRate);
    this.numWeakBoosted = Math.floor(budget * POLICY.weakBoostRate);
  }

  /** encoder 入力の先頭に付ける [speaker cache, FIFO] */
  getEmbeds() {
    return this.cacheEmbeds.concat(this.fifo);
  }

  /**
   * @param {Float32Array[]} inputEmbeds このステップの encoder 入力 ([cache, fifo, chunk, lookahead])
   * @param {Float32Array[]} probs inputEmbeds の各フレームの話者確率 (encoder フレームレートに平均プーリング済み)
   * @param {number} numChunkFrames lookahead を除いたチャンクのフレーム数
   */
  update(inputEmbeds, probs, numChunkFrames) {
    const numCache = this.cacheEmbeds.length;
    const chunkStart = numCache + this.fifo.length;
    let fifo = this.fifo.concat(inputEmbeds.slice(chunkStart, chunkStart + numChunkFrames));

    const numPopped = this.#numPoppedFrames(fifo.length);
    if (numPopped > 0) {
      const fifoProbs = probs.slice(numCache, numCache + fifo.length);
      // 圧縮前のキャッシュは今回のステップで確率を再推定できるが、圧縮後は保存済みの確率しか使えない
      const storedProbs = this.isCompressed ? this.cacheProbs : probs.slice(0, numCache);
      let embeds = this.cacheEmbeds.concat(fifo.slice(0, numPopped));
      let cacheProbs = storedProbs.concat(fifoProbs.slice(0, numPopped));
      fifo = fifo.slice(numPopped);
      if (embeds.length > POLICY.cacheLength) {
        [embeds, cacheProbs] = this.#compress(embeds, cacheProbs);
        this.isCompressed = true;
      }
      this.cacheEmbeds = embeds;
      this.cacheProbs = cacheProbs;
    }
    this.fifo = fifo;
  }

  #numPoppedFrames(numFifoFrames) {
    if (numFifoFrames <= this.fifoLength) return 0;
    return Math.min(Math.max(this.updatePeriod, numFifoFrames - this.fifoLength), numFifoFrames);
  }

  // scores[s][t]: 話者 s のキャッシュとしてフレーム t を残す価値
  #frameScores(probs) {
    const S = this.numSpeakers;
    const N = probs.length;
    const th = POLICY.predictionScoreThreshold;
    const scores = Array.from({ length: S }, () => new Float64Array(N));
    const logComp = new Float64Array(S);
    for (let t = 0; t < N; t++) {
      const p = probs[t];
      let sumLogComp = 0;
      for (let s = 0; s < S; s++) {
        logComp[s] = Math.log(Math.max(1 - p[s], th));
        sumLogComp += logComp[s];
      }
      for (let s = 0; s < S; s++) {
        scores[s][t] = p[s] > 0.5 ? Math.log(Math.max(p[s], th)) - logComp[s] + sumLogComp - LOG_HALF : -Infinity;
      }
    }
    // 正のスコアが十分ある話者は、重なり発話 (発話中だがスコア非正) のフレームを除外する
    for (let s = 0; s < S; s++) {
      let positives = 0;
      for (let t = 0; t < N; t++) if (scores[s][t] > 0) positives++;
      if (positives < this.minPositiveScores) continue;
      for (let t = 0; t < N; t++) if (!(scores[s][t] > 0) && probs[t][s] > 0.5) scores[s][t] = -Infinity;
    }
    return scores;
  }

  #boostTopK(scores, k, amount) {
    for (const row of scores) {
      const order = Array.from(row.keys()).sort((a, b) => row[b] - row[a] || a - b);
      for (let i = 0; i < Math.min(k, order.length); i++) row[order[i]] += amount;
    }
  }

  // 重要度の高い cacheLength フレームを残す。話者ごとにまとまり、話者内では元の順序を保つ。
  #compress(embeds, probs) {
    const S = this.numSpeakers;
    const N = probs.length;
    const cacheLength = POLICY.cacheLength;
    const scores = this.#frameScores(probs);
    for (const row of scores) for (let t = cacheLength; t < N; t++) row[t] += POLICY.latestFramesScoreBoost;
    this.#boostTopK(scores, this.numStrongBoosted, -2 * LOG_HALF);
    this.#boostTopK(scores, this.numWeakBoosted, -LOG_HALF);

    // 各話者に無音スロットを +inf で追加し、話者優先の flat index で上位を選ぶ
    const numScored = N + POLICY.silenceFramesPerSpeaker;
    const candidates = [];
    for (let s = 0; s < S; s++) {
      for (let t = 0; t < numScored; t++) candidates.push({ index: s * numScored + t, score: t < N ? scores[s][t] : Infinity });
    }
    candidates.sort((a, b) => b.score - a.score || a.index - b.index);
    const sentinel = numScored * S;
    const picked = candidates
      .slice(0, cacheLength)
      .map(c => (c.score === -Infinity ? sentinel : c.index))
      .sort((a, b) => a - b);

    const zeroProbs = new Float32Array(S);
    const outEmbeds = [];
    const outProbs = [];
    for (const index of picked) {
      const frame = index === sentinel ? N : Math.min(index % numScored, N);
      outEmbeds.push(frame === N ? this.silenceEmbed : embeds[frame]);
      outProbs.push(frame === N ? zeroProbs : probs[frame]);
    }
    return [outEmbeds, outProbs];
  }
}
