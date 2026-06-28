export class VoiceTurnCommit {
  private utteranceSequence = 0
  private committedSequence = 0
  private activeSequence = 0

  startUtterance(): void {
    this.utteranceSequence += 1
    this.activeSequence = this.utteranceSequence
  }

  finishUtterance(): boolean {
    if (!this.activeSequence || this.activeSequence === this.committedSequence) return false
    this.committedSequence = this.activeSequence
    return true
  }

  reset(): void {
    this.activeSequence = 0
    this.committedSequence = this.utteranceSequence
  }
}
