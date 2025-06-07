# RocketsRNN


In recent years, real-time win probability models have become a valuable tool for interpreting the flow and outcome of sporting events. These models provide fans, analysts, and coaches with dynamic insights into a team’s likelihood of victory as the game progresses, based on contextual in-game features. In this project, we developed a deep learning model to estimate the win probability of the Houston Rockets at each point during an NBA game, using only in-game play-by-play data.

We retrieved our data using the nba_api Python library, an official interface to NBA.com’s stats endpoints. Specifically, we utilized the PlayByPlayV3 endpoint to download event-level data for every game played by the Houston Rockets during the 2023–2024 NBA regular season. Each game contains a chronological list of possessions and actions — such as made shots, rebounds, fouls, and turnovers — along with game clock, team IDs, and scores.

Our problem is framed as a binary sequence classification task: at each point in the game, the model outputs a win probability (between 0 and 1), which estimates whether the Rockets will ultimately win. This problem is both sequential and time-dependent, making it well-suited for recurrent neural networks (RNNs), which are capable of modeling temporal dependencies across sequential data.

We focused on building a win probability model from the perspective of a single team — the Houston Rockets — using a streamlined, interpretable set of features engineered directly from the play-by-play logs. Our goal was to evaluate whether a deep learning model could learn meaningful win dynamics and generate realistic probability curves that reflect in-game momentum swings and score changes.
