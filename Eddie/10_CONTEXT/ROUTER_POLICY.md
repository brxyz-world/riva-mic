# Router Policy (seed)

- id: R_hello
  keys: ["hello eddie", "hey eddie", "yo eddie"]
  replyVariants:
    - "hey — i'm listening."
    - "yo. awake & here."
  replyWeights: [0.6, 0.4]
  tool: null

- id: R_exit
  keys: ["goodbye", "bye eddie"]
  replyVariants:
    - "shutting down. ping me anytime."
    - "eddie signing off."
  replyWeights: [0.5, 0.5]
  tool: null

- id: R_hush
  keys: ["hush", "mute"]
  replyVariants:
    - "muting output, staying aware."
    - "quiet mode on."
  replyWeights: [0.5, 0.5]
  tool: null
