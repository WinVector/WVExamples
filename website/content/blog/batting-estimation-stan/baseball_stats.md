---
title: "Who's the Best Batter? Estimating Probabilities from Unevenly Collected Data"
author: "Nina Zumel"
date: 2026-05-20
tags: ["probabilistic  modeling", "python", "stan", "Bayesian data analysis"]
source: https://github.com/WinVector/WVExamples/tree/main/BattingEstimation_Stan
---

In this article, we look at the problem of estimating and comparing probabilities about a population of subjects from unevenly collected observations. Some examples might include:

* The perceived quality of a movie (how often is a movie positively reviewed) when some movies have far more reviews than others.
* The effectiveness of various ad campaigns, when some compaigns have had more exposure than others.
* The efficacy of a certain medical procedure by hospital, when some hospitals have had more cases than others.

For our specific task, we'll try to estimate the "innate" batting ability (the probability of making a hit when at bat)[^1] of major league baseball players in 2023[^2]. For the sake of this article, we will take this single season of data as everything that we know about these players and their batting statistics.

First, let's take a quick look at the data.

[^1]: By *innate*, I don't mean some kind of "natural-born" ability; a player's batting ability is no doubt honed by training and practice. I merely use the word *innate* to emphasize that the probability of a given player making a hit is not necessarily the same as the *observed* rate at which they made hits during a season.<br>  

[^2]: Data from the [Lahman Baseball Database](https://sabr.org/lahman-database/), currently available from the Society for American Baseball Research (SABR). The [Lahman R package](https://cdalzell.github.io/Lahman/) provides an R interface to the database, as well.<br>


```python
battingf = pd.read_csv(datadir + 'battingstats.tsv', sep = "\t")
battingf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>atbat</th>
      <th>hits</th>
      <th>batting_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>abramcj01</td>
      <td>563</td>
      <td>138</td>
      <td>0.245115</td>
    </tr>
    <tr>
      <th>1</th>
      <td>abreujo02</td>
      <td>540</td>
      <td>128</td>
      <td>0.237037</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abreuwi02</td>
      <td>76</td>
      <td>24</td>
      <td>0.315789</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>643</td>
      <td>217</td>
      <td>0.337481</td>
    </tr>
    <tr>
      <th>4</th>
      <td>adamewi01</td>
      <td>553</td>
      <td>120</td>
      <td>0.216998</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>651</th>
      <td>yoshima02</td>
      <td>537</td>
      <td>155</td>
      <td>0.288641</td>
    </tr>
    <tr>
      <th>652</th>
      <td>youngja02</td>
      <td>43</td>
      <td>8</td>
      <td>0.186047</td>
    </tr>
    <tr>
      <th>653</th>
      <td>youngja03</td>
      <td>107</td>
      <td>27</td>
      <td>0.252336</td>
    </tr>
    <tr>
      <th>654</th>
      <td>zavalse01</td>
      <td>175</td>
      <td>30</td>
      <td>0.171429</td>
    </tr>
    <tr>
      <th>655</th>
      <td>zuninmi01</td>
      <td>124</td>
      <td>22</td>
      <td>0.177419</td>
    </tr>
  </tbody>
</table>
<p>656 rows × 4 columns</p>
</div>

We can calculate some summary statistics, too.

<details>
<summary>Click to see code</summary>

```python
nplayers = battingf.shape[0]
print(f'Population of {nplayers} players.')

mean_ba = battingf['batting_avg'].mean()
std_ba = battingf['batting_avg'].std()
print(f'Mean batting average: {mean_ba:.2f}, standard deviation {std_ba:.2f}')

mean_atbat = battingf['atbat'].mean()
std_atbat = battingf['atbat'].std()
print(f'Mean at bats: {mean_atbat:.2f}, standard deviation {std_atbat:.2f}')
```
</details>

```
    Population of 656 players.
    Mean batting average: 0.23, standard deviation 0.07
    Mean at bats: 250.64, standard deviation 192.45
```

Given this information, how do we estimate players' batting ability? 

You may be tempted to simply use a player's observed batting average as an estimate of their batting skill. One issue with this is that not all players get the same number of at-bats. Let's look at the batting averages for all the players, sorted by their number of times at bat. The horizontal line on the graph represents the mean batting average of the population.
    
![Scatterplot of batting averages vs. times at bat, MLB 2023 season](baseball_stats_5_0.png)
<p class="caption">Batting averages versus times at bat. Dark blue horizontal line represents population mean batting average.</p>    


As you can see, the number of at-bats for players in 2023 varied widely; some players were up hundreds of times, and some fewer than ten times.  For players with a lot of at-bats, their observed batting average is probabably a good estimate of their innate batting ability. But for players with fewer at-bats, their observed batting average is more likely to be an over or under estimate of their ability. 

## Finding the Top 10 Batters

We can make this point dramatically by using our naive batting ability estimates to answer the question, **_Who are the top 10 batters_**?

```python
naive_top10 = battingf.nlargest(10, 'batting_avg')
naive_top10
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>atbat</th>
      <th>hits</th>
      <th>batting_avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>140</th>
      <td>culbech01</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>124</th>
      <td>colliza01</td>
      <td>4</td>
      <td>2</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>325</th>
      <td>lopezal03</td>
      <td>2</td>
      <td>1</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>450</th>
      <td>perezmi03</td>
      <td>8</td>
      <td>4</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>593</th>
      <td>tuckeco01</td>
      <td>8</td>
      <td>4</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>626</th>
      <td>wallfo01</td>
      <td>13</td>
      <td>6</td>
      <td>0.461538</td>
    </tr>
    <tr>
      <th>583</th>
      <td>toroab01</td>
      <td>18</td>
      <td>8</td>
      <td>0.444444</td>
    </tr>
    <tr>
      <th>165</th>
      <td>downsje01</td>
      <td>5</td>
      <td>2</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>229</th>
      <td>graytr01</td>
      <td>5</td>
      <td>2</td>
      <td>0.400000</td>
    </tr>
    <tr>
      <th>121</th>
      <td>clemeer01</td>
      <td>50</td>
      <td>19</td>
      <td>0.380000</td>
    </tr>
  </tbody>
</table>
</div>



Do you trust this ranking? Probably not---notice that most of the players in the top ten using this naive measure have actually been at bat very few times, and their batting averages are unrealistically high. Remember: the average batting average in the league for this season is 0.23, and the standard deviation is small. Batting averages of 1.0 or even 0.5 are highly improbable estimates of actual player ability.

This is analogous to sorting the rankings of a product on an online shopping site[^2_1]. Which assessment would you consider more reliable: 
* one with a five-star average rating calculated from only one or two ratings, 
* or one with a 4.5 star rating calculated from 200 ratings? 

Personally, I would be more likely to trust the assesment of the second product.

Given that our observations of the players are so uneven, is there a better way estimate how good a batter each player really is?

We would like a method that handles players with very few observations in a reasonable way. If a player has been at bat only once, their observed batting average is either 1 or 0: either they look perfect, or they look terrible. Since they are most likely neither, we'd like to assume a reasonable estimate of their batting ability, one we can use while we are waiting for more data. And of course, we want a method where the estimate improves as more data becomes available.

[^2_1]: See Evan Miller's article [How Not To Sort By Average Rating](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html) for an alternative, frequentist solution to the  "sort by rating" problem, in the context of product reviews.<br>

## Estimating Batting Ability with Probabilistic Modeling

One approach that robustly handles players with few at-bats is *probabilistic modeling*. In this article, we will implement a probabilistic model for the batting ability problem using [Stan](https://mc-stan.org/). We won't explain the Stan code in depth, but we will try to explain what the model is doing.

The idea behind this model is that a player's empirical batting performance is merely an *observable approximation* of the player's *unobservable* "innate batting ability," which we'll define as the probability that a batter will make a hit when he or she is at bat. This is illustrated in the figure below.

![Model of Batting Process](batting_model.png)

<p class="caption">Working model of the hit generation process. An individual batter has a given (unobservable) batting ability, drawn from the distribution of batting abilities for the population. For the player's <code>n</code> at-bats, we observe the number of hits and misses during play.</caption>

In order to estimate player batting ability, which we cannot directly observe, we will define a probabilistic process to "explain" the observable batting performance in terms of the unobservable player ability.

Specifically, we'll model each player as a coin, where a hit is "heads", and the number of at-bats is the number of flips. We'll call the (unknown) probability of coming up heads (getting a hit) `gamma`. Then we can model each player as a binomial:

```
      hits_i ~ binomial(atbat_i, gamma_i)
```

In other words, `gamma` is the player's innate batting ability.

We'll further assume that the player `gamma`s are distributed around some (also unknown) "global player batting ability." The idea here is that all the players come from the same population, so their batting performances are somewhat similar. This implies that player batting abilities tend to cluster around some average batting ability, and very high (or low) abilities are unlikely. 

This is *only* an assumption, but we consider it plausible because of real-life observations like "batting averages for professional players tend to be around 0.25ish, and super high batting averages are unlikely"---an observation we can back up with the data.

    
![Distribution of batting averages in the 2023 season](baseball_stats_10_0.png)
<p class="caption">Distribution of observed batting averages for the 2023 MLB season.</caption>


An advantage of probabilistic modeling is that it allows us to incorporate these types of assumptions or domain knowledge into our analysis in a principled way. With more common frequentist analyses, like the above naive approach, we don't have a way to express notions like "player abilities are in a tight, non-uniform distribution," without resorting to ad-hoc rules such as "only consider batters who have more than 100 at-bats."

To continue: we want to model the "global player batting ability" as a distribution from which individual player `gamma`s are drawn. Since the players are binomial, we'll assume that the `gamma`s are distributed as a beta distribution. 

```
      gamma ~ beta(a, b)
```

In Bayesian parlance, the distribution `beta(a, b)` represents the *priors* on `gamma` (player batting ability). For a player with only a few at-bats, there is little information on their individual ability, so the model will estimate that their batting ability is near some average batting ability. For players with many at-bats, the model will have enough information to pull the estimate away from the grand mean.

Intuitively, the  parameters `a` and `b` represent `a` "pseudo-hits" for `a+b` "pseudo-atbats". The larger `a+b` is, the more observations will be required to pull a player's estimated ability away from the grand mean (`a/(a+b)`). In other words, this formulation smooths all the estimated batting averages towards some (estimated) grand mean. The quantity `a+b` specifies the strength of the smoothing.

We can control how much we smooth to the mean, and what the mean is, by explicitly picking `a` and `b`. In this model, however, we will use Stan to estimate `a` and `b` from the data.

Below is the code for the Stan model. Don't worry if you can't read it; the explanation above and the comments in the code should be sufficient.


```python
stan_model_src = """
data {                                               // this block describes the training data
  int<lower=1> n_players;                            // number of players observed
  array[n_players] int<lower=0> hits;                // number of hits - needs to be integer type because of binomial call
  array[n_players] int<lower=0> atbat;               // number of at-bats - needs to be integer type because of binomial call
}
parameters {                                           // this block declares the parameters to be estimated
  vector<lower=0, upper=1>[n_players] gamma;           // unobserved "true" batting abilities
  real<lower=0> a;                                     // pseudo-hits
  real<lower=0> b;                                     // pseudo-misses
}
model {                                             // this block describes the relations between parameters and data
  gamma ~ beta(a, b);                               // distribution of unobservable batting ability
  hits ~ binomial(atbat, gamma);                    // relation of hits to per-player ability
}
"""
```

Unlike most modeling systems, Stan does not return point estimates of the parameters it is trying to fit. It instead uses Monte Carlo sampling to jointly generate sets of parameters (called samples) that are consistent with the training data. Each of these samples (4000 of them, in this case) represents a "possible world" that could generate the observed data. We can use these possible worlds to not only calculate point estimates of the parameters we want, but also uncertainty ranges around those estimates.

Behind the scenes, we have fit the model, and saved Stan's generated samples into a data frame named `fit_Stan`. For all the details, see the source code linked at the top of this article.


```python
fit_Stan.shape
```

```
(4000, 668)
```


## Estimate of the Priors

Let's look at Stan's estimates for `a` and `b`. Do we get reasonable distributions of pseudo-observations and global batting ability? 

As a diagnostic on the model, we would like to see that the distributions of both `a` and `b` are unimodal (which they are, but for brevity the plots are omitted). We'd also like to see that the mean of the beta distribution is near the observed mean batting average in every sample.

<details>
<summary>Click to see code</summary>

```python
## check the a and b estimates. 

abframe = fit_Stan[['a', 'b']].copy()
abframe['pseudo_obsv'] = abframe['a'] + abframe['b']
abframe['global_ba'] = abframe['a']/abframe['pseudo_obsv']
abframe['variance'] = abframe['a']*abframe['b']/( abframe['pseudo_obsv']**2 * (abframe['pseudo_obsv'] + 1) )

print(f"""
      Mean pseudo observation estimate: {abframe['pseudo_obsv'].mean():.3f}; 
      Mean global batting ability estimate: {abframe['global_ba'].mean():.3f}, compared to observed mean batting average {mean_ba:.3f}
""")
```
</details>

```    

Mean pseudo observation estimate: 434.612; 
Mean global batting ability estimate: 0.244, compared to observed mean batting average 0.227

```    

![Distributions of beta means and pseudo-observations from Stan estimates](baseball_stats_20_0.png)
    


The mean of `beta` is generally between 0.24 and 0.25, which is not far from the observed mean batting average of 0.23. The large number of pseudo-observations corresponds to beta distributions with fairly low variance, which is again consistent with our empirical observation. It also means that there will be a lot of smoothing on the estimates.

As a sidenote, it is often common practice to add tight priors to `a` and `b`---that is, to force `a+b` to be small. This is both to keep the priors "uninformative," as in frequentist formulations, and to force the probability estimates to be close to the empirical observations. As we will see later in this article, allowing Stan to pick `a` and `b` such that `a+b` is large, reduced the variance on ability estimates, but in a way that is consistent with the real-world batting performances. Less smoothing on the estimates would have led to too much variance in batting performance.

## Estimating Batting Ability

Now let's get all the samples of player gammas.


```python
batting_estimates = fit_Stan.filter(like='gamma').copy()
batting_estimates.columns = battingf['playerID']
batting_estimates
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>playerID</th>
      <th>abramcj01</th>
      <th>abreujo02</th>
      <th>abreuwi02</th>
      <th>...</th>
      <th>youngja03</th>
      <th>zavalse01</th>
      <th>zuninmi01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.255413</td>
      <td>0.248422</td>
      <td>0.258750</td>
      <td>...</td>
      <td>0.253605</td>
      <td>0.227349</td>
      <td>0.222809</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.244398</td>
      <td>0.217937</td>
      <td>0.251756</td>
      <td>...</td>
      <td>0.283892</td>
      <td>0.217628</td>
      <td>0.237123</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.243953</td>
      <td>0.258032</td>
      <td>0.263147</td>
      <td>...</td>
      <td>0.208346</td>
      <td>0.220289</td>
      <td>0.219931</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.249291</td>
      <td>0.242690</td>
      <td>0.261040</td>
      <td>...</td>
      <td>0.265126</td>
      <td>0.221886</td>
      <td>0.227214</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.237667</td>
      <td>0.251283</td>
      <td>0.240370</td>
      <td>...</td>
      <td>0.233977</td>
      <td>0.219615</td>
      <td>0.237419</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3995</th>
      <td>0.259563</td>
      <td>0.245729</td>
      <td>0.300755</td>
      <td>...</td>
      <td>0.223936</td>
      <td>0.206689</td>
      <td>0.249676</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>0.245689</td>
      <td>0.236702</td>
      <td>0.238916</td>
      <td>...</td>
      <td>0.249938</td>
      <td>0.214311</td>
      <td>0.234321</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>0.224689</td>
      <td>0.227445</td>
      <td>0.234514</td>
      <td>...</td>
      <td>0.274076</td>
      <td>0.252756</td>
      <td>0.234653</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>0.228148</td>
      <td>0.233065</td>
      <td>0.252618</td>
      <td>...</td>
      <td>0.233102</td>
      <td>0.257751</td>
      <td>0.235676</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>0.263306</td>
      <td>0.248453</td>
      <td>0.251055</td>
      <td>...</td>
      <td>0.256680</td>
      <td>0.194360</td>
      <td>0.225084</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 656 columns</p>
</div>
<p class="caption">Each row represents a combination of player batting abilities that is consistent with the training data.</p>



For every player, we can use the above set of Stan estimates to get a point estimate of their gamma (we'll use the mean; you can also use the median), and an uncertainty interval that covers 95% of the Stan estimates.

<details>
<summary>Click to see code</summary>

```python

nplayers = battingf.shape[0]
means = [np.mean(batting_estimates.iloc[:, i]) for i in range(nplayers)]
# calculate the 95% uncertainty interval
intervals = [np.percentile(batting_estimates.iloc[:, i], [2.5, 97.5]) for i in range(nplayers)]
interval_bottom = [interv[0] for interv in intervals]
interval_top = [interv[1] for interv in intervals]

battingf['gamma'] = means
battingf['g_min'] = interval_bottom
battingf['g_max'] = interval_top

battingf

```
</details>


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>atbat</th>
      <th>hits</th>
      <th>batting_avg</th>
      <th>gamma</th>
      <th>g_min</th>
      <th>g_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>abramcj01</td>
      <td>563</td>
      <td>138</td>
      <td>0.245115</td>
      <td>0.244644</td>
      <td>0.217479</td>
      <td>0.271896</td>
    </tr>
    <tr>
      <th>1</th>
      <td>abreujo02</td>
      <td>540</td>
      <td>128</td>
      <td>0.237037</td>
      <td>0.240080</td>
      <td>0.212640</td>
      <td>0.266951</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abreuwi02</td>
      <td>76</td>
      <td>24</td>
      <td>0.315789</td>
      <td>0.254437</td>
      <td>0.215732</td>
      <td>0.295471</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>643</td>
      <td>217</td>
      <td>0.337481</td>
      <td>0.300198</td>
      <td>0.271661</td>
      <td>0.330569</td>
    </tr>
    <tr>
      <th>4</th>
      <td>adamewi01</td>
      <td>553</td>
      <td>120</td>
      <td>0.216998</td>
      <td>0.228769</td>
      <td>0.202451</td>
      <td>0.256487</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>651</th>
      <td>yoshima02</td>
      <td>537</td>
      <td>155</td>
      <td>0.288641</td>
      <td>0.268467</td>
      <td>0.240529</td>
      <td>0.297654</td>
    </tr>
    <tr>
      <th>652</th>
      <td>youngja02</td>
      <td>43</td>
      <td>8</td>
      <td>0.186047</td>
      <td>0.238932</td>
      <td>0.201439</td>
      <td>0.278378</td>
    </tr>
    <tr>
      <th>653</th>
      <td>youngja03</td>
      <td>107</td>
      <td>27</td>
      <td>0.252336</td>
      <td>0.245744</td>
      <td>0.209507</td>
      <td>0.283893</td>
    </tr>
    <tr>
      <th>654</th>
      <td>zavalse01</td>
      <td>175</td>
      <td>30</td>
      <td>0.171429</td>
      <td>0.223203</td>
      <td>0.190155</td>
      <td>0.258194</td>
    </tr>
    <tr>
      <th>655</th>
      <td>zuninmi01</td>
      <td>124</td>
      <td>22</td>
      <td>0.177419</td>
      <td>0.229210</td>
      <td>0.194561</td>
      <td>0.265152</td>
    </tr>
  </tbody>
</table>
<p>656 rows × 7 columns</p>
</div>
<p class="caption">Point estimates of batting abilities, along with upper and lower bounds on 95% uncertainty intervals</p>

### The Top 10 Batters, According to Stan

Here is the roster of top 10 batters, according to the Stan point estimates. Notice that all the players in this roster have been at bat hundreds of times, so we can consider this top 10 to be more trustworthy. 

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>atbat</th>
      <th>hits</th>
      <th>batting_avg</th>
      <th>gamma</th>
      <th>g_min</th>
      <th>g_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>arraelu01</td>
      <td>574</td>
      <td>203</td>
      <td>0.353659</td>
      <td>0.306837</td>
      <td>0.278873</td>
      <td>0.336221</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>643</td>
      <td>217</td>
      <td>0.337481</td>
      <td>0.300198</td>
      <td>0.271661</td>
      <td>0.330569</td>
    </tr>
    <tr>
      <th>200</th>
      <td>freemfr01</td>
      <td>637</td>
      <td>211</td>
      <td>0.331240</td>
      <td>0.296087</td>
      <td>0.268344</td>
      <td>0.324402</td>
    </tr>
    <tr>
      <th>158</th>
      <td>diazya01</td>
      <td>525</td>
      <td>173</td>
      <td>0.329524</td>
      <td>0.291116</td>
      <td>0.262371</td>
      <td>0.321038</td>
    </tr>
    <tr>
      <th>520</th>
      <td>seageco01</td>
      <td>477</td>
      <td>156</td>
      <td>0.327044</td>
      <td>0.287488</td>
      <td>0.258570</td>
      <td>0.318180</td>
    </tr>
    <tr>
      <th>61</th>
      <td>bettsmo01</td>
      <td>584</td>
      <td>179</td>
      <td>0.306507</td>
      <td>0.280108</td>
      <td>0.253704</td>
      <td>0.307470</td>
    </tr>
    <tr>
      <th>62</th>
      <td>bichebo01</td>
      <td>571</td>
      <td>175</td>
      <td>0.306480</td>
      <td>0.279565</td>
      <td>0.252326</td>
      <td>0.307508</td>
    </tr>
    <tr>
      <th>53</th>
      <td>bellico01</td>
      <td>499</td>
      <td>153</td>
      <td>0.306613</td>
      <td>0.277572</td>
      <td>0.249387</td>
      <td>0.306813</td>
    </tr>
    <tr>
      <th>469</th>
      <td>ramirha02</td>
      <td>400</td>
      <td>125</td>
      <td>0.312500</td>
      <td>0.277267</td>
      <td>0.247371</td>
      <td>0.310371</td>
    </tr>
    <tr>
      <th>410</th>
      <td>naylojo01</td>
      <td>452</td>
      <td>139</td>
      <td>0.307522</td>
      <td>0.276918</td>
      <td>0.247745</td>
      <td>0.308113</td>
    </tr>
  </tbody>
</table>
</div>
<p class="caption">The top 10 batters, as given by the Stan point estimates of batter ability</p>


Let's plot the top 10 players' `gamma`s, along with their observed batting average and the estimated global mean batting ability.
    
![Estimated gammas, with 95% uncertainty intervals, for top 10 players](baseball_stats_29_0.png)
<p class="caption">Gamma estimates for the top 10 batters. Gamma and 95% uncertainty intervals in green; observed batting averages in purple. The horizontal dashed line is the estimated mean player batting ability.</p>    


You might wonder why the observed batting averages are consistently so much higher than the estimated batting abilities. This is because the model is smoothing all the estimates towards the priors. Hence, performance estimates for high performing batters will be biased down, and performance estimates for low performing batters will be biased up. This is not a property unique to Stan; it is a property of the smoothing process.

It is also worth pointing out, however, that the uncertainty intervals are away from the estimated global mean (horizontal dashed line), indicating that the model identifies these players as having above average batting ability. Furthermore, the ranking order of the players according to their `gamma` estimates is generally consistent with the ranking of their empirical batting averages.

What about the players with very few at-bats? As desired, their `gamma`s are near the estimated global mean of 0.24.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>atbat</th>
      <th>hits</th>
      <th>batting_avg</th>
      <th>gamma</th>
      <th>g_min</th>
      <th>g_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>46</th>
      <td>barretr01</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243261</td>
      <td>0.204423</td>
      <td>0.284172</td>
    </tr>
    <tr>
      <th>111</th>
      <td>castidi02</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243484</td>
      <td>0.202068</td>
      <td>0.287601</td>
    </tr>
    <tr>
      <th>124</th>
      <td>colliza01</td>
      <td>4</td>
      <td>2</td>
      <td>0.5</td>
      <td>0.246683</td>
      <td>0.206278</td>
      <td>0.288201</td>
    </tr>
    <tr>
      <th>140</th>
      <td>culbech01</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.245780</td>
      <td>0.207254</td>
      <td>0.286393</td>
    </tr>
    <tr>
      <th>165</th>
      <td>downsje01</td>
      <td>5</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.246108</td>
      <td>0.206405</td>
      <td>0.287695</td>
    </tr>
    <tr>
      <th>205</th>
      <td>fulmemi01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243642</td>
      <td>0.205138</td>
      <td>0.284218</td>
    </tr>
    <tr>
      <th>229</th>
      <td>graytr01</td>
      <td>5</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.245800</td>
      <td>0.205371</td>
      <td>0.288453</td>
    </tr>
    <tr>
      <th>242</th>
      <td>hamilbi02</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243317</td>
      <td>0.205493</td>
      <td>0.283204</td>
    </tr>
    <tr>
      <th>243</th>
      <td>hamilca01</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.241302</td>
      <td>0.203104</td>
      <td>0.282519</td>
    </tr>
    <tr>
      <th>290</th>
      <td>kaiseco01</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.241975</td>
      <td>0.202445</td>
      <td>0.282454</td>
    </tr>
    <tr>
      <th>325</th>
      <td>lopezal03</td>
      <td>2</td>
      <td>1</td>
      <td>0.5</td>
      <td>0.245427</td>
      <td>0.205673</td>
      <td>0.287053</td>
    </tr>
    <tr>
      <th>362</th>
      <td>mccoyma01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243813</td>
      <td>0.205295</td>
      <td>0.285935</td>
    </tr>
    <tr>
      <th>386</th>
      <td>millesh01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243820</td>
      <td>0.204505</td>
      <td>0.284791</td>
    </tr>
    <tr>
      <th>388</th>
      <td>mitchca01</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.241544</td>
      <td>0.201192</td>
      <td>0.284165</td>
    </tr>
    <tr>
      <th>424</th>
      <td>okeych01</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.242944</td>
      <td>0.202513</td>
      <td>0.285449</td>
    </tr>
    <tr>
      <th>482</th>
      <td>reynoma03</td>
      <td>5</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.243430</td>
      <td>0.204012</td>
      <td>0.285005</td>
    </tr>
    <tr>
      <th>514</th>
      <td>sborzjo01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243737</td>
      <td>0.203860</td>
      <td>0.285506</td>
    </tr>
    <tr>
      <th>521</th>
      <td>seaglch01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243371</td>
      <td>0.203613</td>
      <td>0.285376</td>
    </tr>
    <tr>
      <th>527</th>
      <td>shewmbr01</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.241642</td>
      <td>0.200795</td>
      <td>0.284840</td>
    </tr>
    <tr>
      <th>529</th>
      <td>sianimi01</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.241498</td>
      <td>0.201679</td>
      <td>0.283838</td>
    </tr>
    <tr>
      <th>614</th>
      <td>vilorme01</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.242547</td>
      <td>0.202917</td>
      <td>0.283509</td>
    </tr>
    <tr>
      <th>622</th>
      <td>wainwad01</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.242655</td>
      <td>0.203335</td>
      <td>0.284259</td>
    </tr>
  </tbody>
</table>
</div>
<p class="caption">Observed batting averages and estimated batting ability for players with five or fewer at-bats.</p>


## Who's the best player? Another way to choose

If our goal is in fact to choose the player(s) with the highest batting ability, there is another way to do it, using the "possible worlds" sampled by Stan. For each one of the 4000 possible worlds, we identify the highest ability player. The "true" best player is most likely the one who is best in the most possible worlds.

Below, we find the best player in every Stan sample, and pick our top 10 accordingly. We could of course pick the top 10 in each possible world, and draw our "most likely top 10" from the resulting sets, but picking the single best is easier to code, and gets the point across.

<details>
<summary>Click to see code</summary>

```python
# get the best performance in each sample world
best_perf = batting_estimates.max(axis=1)

# mark which player was the best in each world. ties ok
is_best = batting_estimates.eq(best_perf, axis=0).astype(int) 

# compute the series and convert it to a data frame
mean_best = is_best.mean().reset_index() 
mean_best.columns = ['playerID', 'frac_as_best']

# join it into battingf
battingf = battingf.merge(mean_best, on='playerID')

# top 10 by fraction best
top10_by_frac = battingf.nlargest(10, 'frac_as_best')
top10_by_frac[['playerID', 'frac_as_best', 'batting_avg', 'gamma']]
```
</details>

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>frac_as_best</th>
      <th>batting_avg</th>
      <th>gamma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>arraelu01</td>
      <td>0.33250</td>
      <td>0.353659</td>
      <td>0.306837</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>0.18100</td>
      <td>0.337481</td>
      <td>0.300198</td>
    </tr>
    <tr>
      <th>200</th>
      <td>freemfr01</td>
      <td>0.11000</td>
      <td>0.331240</td>
      <td>0.296087</td>
    </tr>
    <tr>
      <th>158</th>
      <td>diazya01</td>
      <td>0.06425</td>
      <td>0.329524</td>
      <td>0.291116</td>
    </tr>
    <tr>
      <th>520</th>
      <td>seageco01</td>
      <td>0.04150</td>
      <td>0.327044</td>
      <td>0.287488</td>
    </tr>
    <tr>
      <th>469</th>
      <td>ramirha02</td>
      <td>0.01650</td>
      <td>0.312500</td>
      <td>0.277267</td>
    </tr>
    <tr>
      <th>62</th>
      <td>bichebo01</td>
      <td>0.01250</td>
      <td>0.306480</td>
      <td>0.279565</td>
    </tr>
    <tr>
      <th>410</th>
      <td>naylojo01</td>
      <td>0.01075</td>
      <td>0.307522</td>
      <td>0.276918</td>
    </tr>
    <tr>
      <th>61</th>
      <td>bettsmo01</td>
      <td>0.01050</td>
      <td>0.306507</td>
      <td>0.280108</td>
    </tr>
    <tr>
      <th>53</th>
      <td>bellico01</td>
      <td>0.00975</td>
      <td>0.306613</td>
      <td>0.277572</td>
    </tr>
  </tbody>
</table>
</div>
<p class="caption">Too 10 batters, calculated by how often each player ranked best in a Stan sample.</p>


This is substantially the same set of players as were selected by looking just at the point estimates. That's good! It gives us confidence that these are indeed the players with the highest batting ability.

Let's mark the players who show up in the top 10, by either criterion. We'll also check for differences in the two top 10 sets.

<details>
<summary>Click to see code</summary>

```python
top10_by_frac = set(battingf.nlargest(10, 'frac_as_best')['playerID'])
top10_by_gamma = set(battingf.nlargest(10, 'gamma')['playerID'])

# take a look at the differences in the sets
print(f'Picked by point estimate but not by fraction best: { top10_by_gamma.difference(top10_by_frac) }')
print(f'Picked by fraction best but not by point estimate: { top10_by_frac.difference(top10_by_gamma) }')

in_top10_set =  top10_by_gamma.union(top10_by_frac)
in_top10 = battingf['playerID'].isin(in_top10_set)

battingf['in_top10'] = in_top10.astype(str)
battingf.loc[battingf['in_top10']=='True', ['playerID', 'atbat', 'hits', 'batting_avg', 'gamma', 'frac_as_best']]
```
</details>

```
    Picked by point estimate but not by fraction best: set()
    Picked by fraction best but not by point estimate: set()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>atbat</th>
      <th>hits</th>
      <th>batting_avg</th>
      <th>gamma</th>
      <th>frac_as_best</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>643</td>
      <td>217</td>
      <td>0.337481</td>
      <td>0.300198</td>
      <td>0.18100</td>
    </tr>
    <tr>
      <th>34</th>
      <td>arraelu01</td>
      <td>574</td>
      <td>203</td>
      <td>0.353659</td>
      <td>0.306837</td>
      <td>0.33250</td>
    </tr>
    <tr>
      <th>53</th>
      <td>bellico01</td>
      <td>499</td>
      <td>153</td>
      <td>0.306613</td>
      <td>0.277572</td>
      <td>0.00975</td>
    </tr>
    <tr>
      <th>61</th>
      <td>bettsmo01</td>
      <td>584</td>
      <td>179</td>
      <td>0.306507</td>
      <td>0.280108</td>
      <td>0.01050</td>
    </tr>
    <tr>
      <th>62</th>
      <td>bichebo01</td>
      <td>571</td>
      <td>175</td>
      <td>0.306480</td>
      <td>0.279565</td>
      <td>0.01250</td>
    </tr>
    <tr>
      <th>158</th>
      <td>diazya01</td>
      <td>525</td>
      <td>173</td>
      <td>0.329524</td>
      <td>0.291116</td>
      <td>0.06425</td>
    </tr>
    <tr>
      <th>200</th>
      <td>freemfr01</td>
      <td>637</td>
      <td>211</td>
      <td>0.331240</td>
      <td>0.296087</td>
      <td>0.11000</td>
    </tr>
    <tr>
      <th>410</th>
      <td>naylojo01</td>
      <td>452</td>
      <td>139</td>
      <td>0.307522</td>
      <td>0.276918</td>
      <td>0.01075</td>
    </tr>
    <tr>
      <th>469</th>
      <td>ramirha02</td>
      <td>400</td>
      <td>125</td>
      <td>0.312500</td>
      <td>0.277267</td>
      <td>0.01650</td>
    </tr>
    <tr>
      <th>520</th>
      <td>seageco01</td>
      <td>477</td>
      <td>156</td>
      <td>0.327044</td>
      <td>0.287488</td>
      <td>0.04150</td>
    </tr>
  </tbody>
</table>
</div>
<p class="caption">Players marked as belonging to top 10 by either ranking criterion.</p>


### Comparing the Naive Ranking to Stan's Ranking

Now let's plot all the players, sorted by observed batting average (lowest to highest). We'll plot the estimated player ability (`gamma`), along with the 95% uncertainty intervals around the estimates (in light gray). The points are also color coded by whether or not the player made at least one of the top 10 lists (in green) or not (in purple). The dashed line is the estimated mean player ability.

    
![Estimated player abilities and 95% uncertainty intervals. Players sorted by observed batting average (lowest to highest)](baseball_stats_39_0.png)
<p class="caption">Estimated player abilities, with players sorted by observed batting average (lowest to highest). Green points indicate top 10 batters. 95% uncertainty intervals on ability estimates shown in gray. Dashed line represents estimated mean player ability. Right click on image to get full size version.</p>


There are a few things to note in this graph. First, the ranking of ability estimates roughly correlate with the ranking of observed batting averages, as desired. There is a cluster of purple players to the right of the green players who have relatively low (actually, average) estimated batting ability, but high observed batting average. These are the players who weren't at bat very often, but who were successful when they were. The model did not have enough data on these players to move the ability estimates away from the prior. Similarly, there are players at the far left of the graph who cluster around the mean, even though their observed batting averages are zero, or nearly so. These are also players with only a few at bats, so their estimated abilities still smooth strongly into the prior.

So if the goal of estimating player ability is to identify the best players, then this model has been able to do so. It automatically discounts spurious empirical estimates that are likely inaccurate due to insufficient data, without the analyst having to specify what "insufficient data" is in an ad-hoc way. It also provides reasonable assumptions about the abilities of low information (low at-bat) players---assumptions that are based on the population data.

## Comparing the Model to Reality

Here are the three players who were most often ranked best in a Stan sample.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>frac_as_best</th>
      <th>batting_avg</th>
      <th>gamma</th>
      <th>career batting average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>arraelu01</td>
      <td>0.3325</td>
      <td>0.353659</td>
      <td>0.306837</td>
      <td>0.317</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>0.1810</td>
      <td>0.337481</td>
      <td>0.300198</td>
      <td>0.288</td>
    </tr>
    <tr>
      <th>200</th>
      <td>freemfr01</td>
      <td>0.1100</td>
      <td>0.331240</td>
      <td>0.296087</td>
      <td>0.299</td>
    </tr>
  </tbody>
</table>
</div>
<p class="caption">Top Three Players, probability of being best player, observed 2023 batting average, estimated batting ability, and career batting average as of May 2026.</p>


The top player, `arraelu01`, is best-ranked by both "probability of being best" and point estimate criteria. He is Venezuelan-born infielder, Luis Arráez. Here's what his [Wikipedia page](https://en.wikipedia.org/wiki/Luis_Arr%C3%A1ez) has to say about him:

> Known for his ability to put the ball in play and not striking out, Arráez is considered one of the best contact hitters of his generation. From 2022 to 2024, Arráez became the first player in MLB history to win three consecutive batting titles with three different teams.... He was also the second player in the modern era to win a batting title in each league and the first to do so in consecutive years. 

Note that his career MLB batting average[^3] (calculated from 2019 through May 17, 2026) is 0.317. This is pretty close to our estimated `gamma` of 0.307. In fact, our estimate is a better prediction of Arráez's career performance (so far) than the simple observation of his 2023 season batting average is---even though we estimated his ability using only this single season!

The next two players (as ranked by probability of being best) are Ronald Acuña, Jr. and Freddie Freeman. Note that for all these players, both Stan's batting ability estimate and player career batting average are below their observed 2023 season batting averages---showing that smoothing performance estimates to the population mean was a reasonable modeling choice. Note also that the career batting averages for these top three players also fell within the 95% uncertainty intervals of ability, as estimated by Stan. 


[^3]: All career batting averages as given by Wikipedia on May 19, 2026.<br>

## Matching Summaries

Let's further compare Stan's batting ability estimates with observations from the data.

<details>
<summary>Click to see code</summary>

```python
mean_ability = battingf['gamma'].mean()
std_ability = battingf['gamma'].std()

print(f'Mean observed batting average: {mean_ba:.3f}, standard deviation {std_ba:.3f}.')
print(f'Mean estimated batting ability: {mean_ability:.3f}, standard deviation {std_ability:.3f}.')
```
</details>

```
Mean observed batting average: 0.227, standard deviation 0.075.
Mean estimated batting ability: 0.244, standard deviation 0.012.
```

As we saw previously, Stan's estimated mean batter ability is close to what was observed in the data, but the standard deviation of the ability estimates is much lower!  

This is not surprising: we also know that the number of player at-bats varied widely, and players with few at-bats will have observed batting averages that will tend to over- or under- estimate their actual abilities. This is why **observed batting average standard deviation is so much higher than estimated batting ability standard deviation**.

In order to properly compare Stan's ability estimates to actual observations, we have to simulate the season in each Stan sample. That is, in each possible world, we give each player the same number of at-bats as they had in 2023, and generate a plausible observed batting average, given that number of at-bats. This is shown below.

<details>
<summary>Click to see code</summary>

```python
def draw_synthetic_hitrate (atbat, bavec):
    return rng.binomial(atbat, bavec)/atbat

synthetic_hitrate_frame = pd.DataFrame({
    col: draw_synthetic_hitrate(battingf['atbat'][i], batting_estimates[col])
    for i, col in enumerate(batting_estimates.columns)
})

synthetic_hitrate_frame
```
</details>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>abramcj01</th>
      <th>abreujo02</th>
      <th>abreuwi02</th>
      <th>...</th>
      <th>youngja03</th>
      <th>zavalse01</th>
      <th>zuninmi01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.243339</td>
      <td>0.246296</td>
      <td>0.381579</td>
      <td>...</td>
      <td>0.177570</td>
      <td>0.245714</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.257549</td>
      <td>0.209259</td>
      <td>0.368421</td>
      <td>...</td>
      <td>0.261682</td>
      <td>0.228571</td>
      <td>0.169355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.238011</td>
      <td>0.262963</td>
      <td>0.210526</td>
      <td>...</td>
      <td>0.224299</td>
      <td>0.200000</td>
      <td>0.266129</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.282416</td>
      <td>0.253704</td>
      <td>0.276316</td>
      <td>...</td>
      <td>0.214953</td>
      <td>0.291429</td>
      <td>0.225806</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.218472</td>
      <td>0.237037</td>
      <td>0.236842</td>
      <td>...</td>
      <td>0.214953</td>
      <td>0.211429</td>
      <td>0.209677</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3995</th>
      <td>0.289520</td>
      <td>0.227778</td>
      <td>0.421053</td>
      <td>...</td>
      <td>0.224299</td>
      <td>0.234286</td>
      <td>0.241935</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>0.253996</td>
      <td>0.229630</td>
      <td>0.381579</td>
      <td>...</td>
      <td>0.233645</td>
      <td>0.217143</td>
      <td>0.225806</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>0.236234</td>
      <td>0.216667</td>
      <td>0.250000</td>
      <td>...</td>
      <td>0.252336</td>
      <td>0.188571</td>
      <td>0.241935</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>0.275311</td>
      <td>0.251852</td>
      <td>0.236842</td>
      <td>...</td>
      <td>0.196262</td>
      <td>0.360000</td>
      <td>0.266129</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>0.291297</td>
      <td>0.244444</td>
      <td>0.263158</td>
      <td>...</td>
      <td>0.271028</td>
      <td>0.165714</td>
      <td>0.250000</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 656 columns</p>
</div>
<p class="caption">Each row represents synthetic batting statistics for the 2023 season, given each player's hypothesized batting ability in that "possible world." Eash player's at-bats is the same as in the actual 2023 season.</p>

From these synthetic replays of 2023, we can estimate plausible means and standard deviations of observed batting average.

<details>
<summary>Click to see code</summary>

```python
# get the mean and standard devation on ability for each sample world
mean_synth_vec = synthetic_hitrate_frame.mean(axis=1)
std_synth_vec = synthetic_hitrate_frame.std(axis=1)

# get the average mean and standard deviation over all sample worlds.
mean_synth = mean_synth_vec.mean()
std_synth = std_synth_vec.mean()

print(f'Mean observed batting average: {mean_ba:.3f}, standard deviation {std_ba:.3f}.')
print(f'Mean synthetic batting average observations: {mean_synth:.3f}, standard deviation {std_synth:.3f}.')

```
</details>

```
Mean observed batting average: 0.227, standard deviation 0.075.
Mean synthetic batting average observations: 0.244, standard deviation 0.078.
```

This is much closer to what was actually observed! We can also plot the distribution of batting average standard deviations in each synthetic season, and compare them to the actual observed batting average standard deviaion.
    
![Distribution of synthetic standard deviations](baseball_stats_48_0.png)
<p class="caption">The distribution of batting-average standard deviations, with their mean, in dark blue. The gray dashed line is the actual batting-average standard deviation.</p>

Once we simulate the at-bats, the behaviors in the synthetic worlds are consistent with what was observed in the actual data: observed batting averages vary more widely than innate batting abilities. This also gives us confidence that our model is a reasonable approximation of the real world baseball hit generation process. Specifically, it's an approximation we can use to answer the questions we want to ask, like "who are the best batters?".

## Estimate What You Want to Know, Not Just What You Can Observe

As we've seen in the above example, an advantage of probabilistic modeling is that the analyst is able to distinguish between *observations* and (potentially unobservable) *quantities of interest*. If you, the analyst, can describe a probabilistic process that relates **_what you can see_** to **_what you actually need to know_**, then probabilistic modeling programs like Stan can estimate these quantities for you. 

By specifying the process to describe your problem and your task goal, you can add in prior knowledge or assumptions about the domain in a principled, documentable way, without having to resort to ad-hoc tweaks or data processing. 

In addition, probabilistic modeling systems that are based on Monte Carlo sampling (like Stan) provide samples of "possible worlds" that are consistent with the training data. You can use these samples not only to calculate point estimates of quantities of interest, but also uncertainty intervals around those estimates. You can also use the possible worlds to run simulations and scenarios (like, "who are the top 10 players in each possible world?") to further help you in decision-making. 

Of course, the estimates can only be as good as the process that you describe. But as your understanding of and intuitions about these processes improve, then so, too, can your model. 
