# Router Policy (seed)

- id: R_hello
  mood: inherit
  keys: ["hello eddie", "hey eddie", "yo eddie"]
  replyVariants:
    - "hey - i'm listening."
    - "yo. awake & here."
  replyWeights: [0.6, 0.4]
  weightsByMood:
    upbeat: [0.8, 0.2]
    mellow: [0.4, 0.6]
  moodNotes: "Favors warmer greet when mood=upbeat; leans to calm opener when mellow."
  tool: null

- id: R_exit
  mood: forced
  forcedMood: mellow
  keys: ["goodbye", "bye eddie"]
  replyVariants:
    - "shutting down. ping me anytime."
    - "eddie signing off."
  replyWeights: [0.5, 0.5]
  weightsByMood:
    neutral: [0.6, 0.4]
    mellow: [0.4, 0.6]
  moodNotes: "Force mellow so exit lands gently regardless of prior state."
  tool: null

- id: R_hush
  mood: inherit
  keys: ["hush", "mute"]
  replyVariants:
    - "muting output, staying aware."
    - "quiet mode on."
  replyWeights: [0.5, 0.5]
  weightsByMood:
    focused: [0.7, 0.3]
    mellow: [0.3, 0.7]
  moodNotes: "Focused bias keeps confirmations crisp during production blocks."
  tool: null