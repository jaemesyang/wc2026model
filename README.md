# wc2026-model

Predicts the 2026 World Cup with a Poisson model fit on ~49,000 international matches since 1872. Each team has an attack and defense strength learned by maximum likelihood, with exponential time-decay so recent results count more than old ones. That gives scoreline probabilities for any matchup, which I run through 20,000 Monte Carlo simulations to get advancement and title odds. An Elo model runs alongside as a baseline.

Scored against real results, the model hit a Brier score of 0.560 across the 72-match group stage, beating the 0.667 coin-flip baseline.

## How it did

Final tournament Brier: 0.534 across all 103 matches, beating the 0.667 coin flip baseline. Knockout rounds scored better than group stage, since there are usually less upsets in the knockout stage. The actual champion was Spain while we predicted Argentina, though the model predicted Spain reaching the semifinals. 80% of the Round of 32 were called correctly. 

The two biggest busts were Brazil and Colombia, semi-finalist and finalist, respectively. These highlight the models week spot, with no player-level data, it overrrates and inflates teams' ratings that have strong results but thninger squad. This particularly affected a lot of South America sides, and it failed exactly as predicted. 


## Predicted Champion: Argentina

```
— Round of 32 —
  M73    Canada (73%)        def. South Africa
  M74    Germany (62%)       def. Paraguay
  M75    Morocco (55%)       def. Netherlands
  M76    Brazil (58%)        def. Japan
  M77    France (80%)        def. Sweden
  M78    Norway (52%)        def. Ivory Coast
  M79    Ecuador (57%)       def. Mexico
  M80    England (73%)       def. DR Congo
  M81    United States (80%) def. Bosnia and Herzegovina
  M82    Belgium (59%)       def. Senegal
  M83    Colombia (86%)      def. Ghana
  M84    Spain (80%)         def. Austria
  M85    Switzerland (50%)   def. Algeria
  M86    Argentina (92%)     def. Cape Verde
  M87    Portugal (66%)      def. Croatia
  M88    Australia (63%)     def. Egypt

— Round of 16 —
  M89    France (62%)        def. Germany
  M90    Morocco (53%)       def. Canada
  M91    Brazil (81%)        def. Norway
  M92    England (57%)       def. Ecuador
  M93    Colombia (63%)      def. Portugal
  M94    Belgium (52%)       def. United States
  M95    Argentina (77%)     def. Australia
  M96    Spain (80%)         def. Switzerland

— Quarterfinals —
  M97    France (60%)        def. Morocco
  M98    Colombia (70%)      def. Belgium
  M99    Brazil (60%)        def. England
  M100   Argentina (60%)     def. Spain

— Semifinals —
  M101   Colombia (61%)      def. France
  M102   Argentina (64%)     def. Brazil

— Final —
  Argentina (63%)            def. Colombia
```


