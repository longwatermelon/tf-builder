# Idea

I'm thinking of a game where you have to build a transformer by hand to make it behave a certain way, like specified by what outputs should be given by what inputs for example, and you get modules like embedding and mha and mlp and linear (which you can use for unembedding) and softmax and all of that. The user edits the relevant weights (W_Q, W_K, W_V, and W_O if mha for a transformer block, W_E and W_P for embeddings (we want to give the player the option to embed both per-token and positional info), W,b for linear, you know the deal).

We'd have various puzzles with difficulties either easy, medium, or hard, as well as hard-coded "canonical" solutions to the problems that the user can reveal if they can't solve it. If the argmaxes of the output softmax distributions per-token are correct where the correct logit's predicted prob is higher than the runner-up by some small epsilon, then the puzzle is counted as solved. There are three types of solved as well (let n be the number of parameters the user's solution used):
  * Excellent: n <= (# parameters used in the canonical solution)
  * Good: n < (# parameters used in the minimal hard-coded solution)
  * Bad: any other value of n

Generally the puzzles should keep the number of required parameters to edit in a reasonable range. Some tedium should be permitted here and there in order to avoid making tedium a heuristic for "there has to exist a better solution to this puzzle", but we also don't want to make the game just tedious and annoying to play.

I have a git submodule ./nn-builder here as an example game that I made for simple fnns you can take a look through for reference on the kind of game style I'm thinking of.

