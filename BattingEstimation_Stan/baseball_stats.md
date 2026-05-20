```python
import json
import logging
import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel
from plotnine import *
from sklearn import linear_model

rng = np.random.default_rng(2026)

# quiet down Stan
logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())

# set plot size
# plotnine.options.figure_size = (16, 8)

# seed pseudo-rng for repeatability of data generation
# Stan uses its own seeds and state
rng = np.random.default_rng(2025)

# define directories
datadir = "data/"
standir = "stan_models/"
stan_datadir = "stan_data/"

```

# Who's the best batter? Estimating probabilities with unevenly collected data

In this article, we look at the problem of estimating the success rates of a population of subjects (and comparing these success rates), when the observations of these subjects have been collected unevenly. Some examples might include:

* The perceived quality of a movie (how often is a movie positively reviewed) when some movies have far more reviews than others
* The conversion rate of various ad campaigns, when some compaigns have had more exposure than others
* The success rates of a certain medical procedure by hospital, when some hospital has had more cases than others

For our specific task, we'll try to estimate the "innate" batting ability (the probability of making a hit when at bat)[^1] of major league baseball players in 2023 ([SABR Database](https://sabr.org/lahman-database/); [R Interface to SABR](https://cdalzell.github.io/Lahman/)). For the sake of this article, we will take this single season of data as everything that we know about these players and their batting statistics.

First, let's take a quick look at the data.

[^1]: By *innate*, I don't mean some kind of "natural-born" ability; a player's batting ability is no doubt honed by training and practice. I merely use the word *innate* to underline that the probability of a given player making a hit when at bat is not necessarily the same as the *observed* rate at which they made hits during a season.


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

    Population of 656 players.
    Mean batting average: 0.23, standard deviation 0.07
    Mean at bats: 250.64, standard deviation 192.45


Given this information, how do we estimate players' batting ability? 

You may be tempted to simply use a player's observed batting average as an estimate of their batting skill. One issue with this is that not all players get the same number of at-bats. Let's look at the batting averages for all the players, sorted by their number of times at bat. The horizontal line on the graph represents the mean batting average of the population.


```python
mean_ba = np.mean(battingf['batting_avg'])
(
    ggplot(battingf, aes(x="atbat", y="batting_avg")) +
    geom_point() + 
    geom_hline(yintercept = mean_ba, color="darkblue") + 
    scale_x_continuous(name = "times at bat") +  
    ggtitle("Batting Averages, 2023")
)
```


    
![png](baseball_stats_files/baseball_stats_5_0.png)
    


As you can see, the number of at-bats for players in 2023 varied widely; some players were up hundreds of times, and some fewer than ten times. 
For players with a lot of at-bats, their observed batting average is probabably a good estimate of their innate batting ability. But for players with fewer at-bats, their observed batting average is more likely to be an over or under estimate of their ability. 

We can make this point dramatically by trying to use our naive batting ability estimates to ask the question, **_Who are the top 10 batters_**?


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

This is analogous to sorting the rankings of a product on an online shopping site. Which assessment would you consider more reliable: one with a five-star average rating calculated from only one or two ratings, or one with a 4.5 star rating calculated from 200 ratings? Personally, I would be more likely to trust the assesment of the second product.

Given that our observations of the players are so uneven, is there a better way estimate how good a batter each player really is?

We would like a method that handles players with very few observations in a reasonable way. If a player has been at bat only once, their observed batting average is either 1 or 0; either they look perfect, or they look terrible. Since they are most likely neither, we'd like to assume a reasonable estimate of their batting ability, one we can use while we are waiting for more data. And of course, we want a method where the estimate improves as more data becomes available.

## Estimating batting ability with probabilistic modeling

One approach that robustly handles players with few at-bats is *probabilistic modeling*. In this article, we will implement a probabilistic model for the batting ability problem using [Stan](https://mc-stan.org/). We won't explain the Stan code in depth, but we will try to explain what the model is doing.

The idea behind this model is that a player's empirical batting performance is merely an *observable approximation* of the player's *unobservable* "innate batting ability," which we'll define as the probability that a batter will make a hit when he or she is at bat. This is illustrated in the figure below.

![Model of Batting Process](batting_model.png)

**Caption: Our model of the batting process. An individual batter has a given (unobservable) batting ability, drawn from the distribution of batting abilities for the population. For the player's `n` at-bats, we observe the number of hits and misses during play.**

In order to estimate player batting ability, which we cannot directly observe, we will define a probabilistic process to "explain" the observable batting performance in terms of the unobservable player ability.

Specifically, we'll model each player as a coin, where a hit is "heads", and the number of at-bats is the number of flips. We'll call the (unknown) probability of coming up heads (getting a hit) `gamma`. Then we can model each player as a binomial:

```
hits_i ~ binomial(atbat_i, gamma_i)
```

In other words, `gamma` is the player's innate batting ability.

We'll further assume that the player `gamma`s are distributed around some (also unknown) "global player batting ability." The idea here is that all the players in some sense come from the same population, so their batting performances are somewhat similar. This implies that player batting abilities tend to cluster around some average batting ability, and very high (or low) abilities are unlikely. 

This is *only* an assumption, but we consider it plausible because of real-life observations like "batting averages for professional players tend to be around 0.25ish, and super high batting averages are unlikely"---an observation we can back up with the data.


```python
(
    ggplot(battingf, aes(x="batting_avg")) + 
    geom_density() + geom_vline(xintercept=mean_ba, color="darkblue") + 
    ggtitle(f"Distribution of batting averages, mean = {mean_ba:.2f}")
)
```


    
![png](baseball_stats_files/baseball_stats_10_0.png)
    


An advantage of probabilistic modeling is that it allows us to express such assumptions (or other plausible ones) and incorporate them into our analysis in a principled way. With more common frequentist analyses (like our naive approach), we don't have a way to express notions like "player abilities are in a tight, not uniform distribution," without having to resort to ad-hoc rules such as "only consider batters who have more than 100 at-bats."

To continue: we want to model the "global player batting ability" as a distribution from which individual player gammas are drawn. Since the players are binomial, we'll assume that the gammas are distributed as a beta distribution. 

```
gamma ~ beta(a, b); 
```

In Bayesian parlance, the distribution `beta(a, b)` represents the *priors* on `gamma` (player batting ability). For a player with only a few at-bats, there is little information on their individual ability, so the model will estimate that their batting ability is near some average batting ability. For players with many at-bats, the model will have enough information to pull the estimate away from the grand mean.

Intuitively, the  parameters `a` and `b` represent `a` "pseudo-hits" for `a + b` "pseudo-atbats". The larger `a + b` is, the more observations will be required to pull a player's estimated ability away from the 
grand mean (`a/(a+b)`). In other words, this formulation smooths all the estimated batting averages towards some (estimated) grand mean. The quantity `a+b` specifies the strength of the smoothing.

We can control how much we smooth to the mean, and what the mean is, by explicitly picking `a+b`. In this model, however, we will use Stan to estimate `a` and `b` from the data.

Below is the code for the Stan model. Don't worry if you can't read it; the explanation above and the comments in the code should be sufficient.


```python
stan_model_src = """
data {
  int<lower=1> n_players;                            // number of players observed
  array[n_players] int<lower=0> hits;                // number of hits - needs to be integer type because of binomial call
  array[n_players] int<lower=0> atbat;               // number of at-bats - needs to be integer type because of binomial call
}
parameters {
  vector<lower=0, upper=1>[n_players] gamma;           // unobserved "true" batting abilities
  real<lower=0> a;                                     // pseudo-hits
  real<lower=0> b;                                     // pseudo-misses
}
model {
  // relations between parameters and data
  gamma ~ beta(a, b);                               // distribution of unobservable batting ability
  hits ~ binomial(atbat, gamma);                    // relation of hits to per-player ability
}
"""
stan_file_name: str = standir + "batting_model.stan"
with open(stan_file_name, "w", encoding="utf8") as file:
    file.write(stan_model_src)
```


```python
# build up Stan data

stan_data = {
    'n_players': battingf.shape[0],
    'hits': list(battingf['hits']),
    'atbat': list(battingf['atbat']),
}
data_file_name: str = stan_datadir + "batting_model.data.json"
with open(data_file_name, "w", encoding="utf8") as file:
    json.dump(stan_data, file)
```


```python
model = CmdStanModel(stan_file=stan_file_name)
fit = model.sample(
    data=data_file_name,
    iter_warmup=1000,
    iter_sampling=1000,
    show_progress=True,
    show_console=False,
)
# get the samples
fit_Stan = fit.draws_pd().reset_index(drop=True, inplace=False)  # force copy just in case
```


    chain 1 |          | 00:00 Status



    chain 2 |          | 00:00 Status



    chain 3 |          | 00:00 Status



    chain 4 |          | 00:00 Status


                                                                                                                                                                                                                                                                                                                                    


Unlike most modeling systems, Stan does not return point estimates of the parameters it is trying to fit. It instead uses Monte Carlo sampling to jointly generate sets of parameters (called samples) that are consistent with the training data. Each of these samples (4000 of them, in this case) represents a "possible world" that could generate the observed data. We can use these possible worlds to not only calculate point estimates of the parameters we want, but also uncertainty ranges around those estimates.

Behind the scenes, we have fit the model, and saved Stan's generated samples into a data frame called `fit_Stan`.


```python
fit_Stan.shape
```




    (4000, 668)



## Estimate of the priors

Let's look at Stan's estimates for `a` and `b`. Do we get reasonable distributions of pseudo-observations and global batting ability?
How close are the batting ability estimates to the observed mean batting average?

As a diagnostic on the model, we would like to see that the distributions of both `a` and `b` are unimodal (which they are, but for brevity the plots are omitted). We'd also like to see that the mean of the beta distribution is near the observed mean batting average in every sample.


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

    
          Mean pseudo observation estimate: 427.296; 
          Mean global batting ability estimate: 0.244, compared to observed mean batting average 0.227
    



```python
ggplot(abframe, aes(x="global_ba")) + geom_density() + ggtitle("distribution of global batting average estimate")
```


    
![png](baseball_stats_files/baseball_stats_19_0.png)
    


The mean of beta is generally around 0.24, which is close to the observed mean batting average of 0.23. 
The large number of pseudo-observations corresponds to beta distributions with fairly low variance, which is again consistent with our empirical observation. It also means that there will be a lot of smoothing on the estimates.

## Estimating batting ability

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
      <th>acunaro01</th>
      <th>adamewi01</th>
      <th>adamsjo03</th>
      <th>adamsri03</th>
      <th>adelljo01</th>
      <th>adriaeh01</th>
      <th>aguilje01</th>
      <th>...</th>
      <th>wongko01</th>
      <th>wynnsau01</th>
      <th>yastrmi01</th>
      <th>yelicch01</th>
      <th>yepezju01</th>
      <th>yoshima02</th>
      <th>youngja02</th>
      <th>youngja03</th>
      <th>zavalse01</th>
      <th>zuninmi01</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.236159</td>
      <td>0.209769</td>
      <td>0.252895</td>
      <td>0.331512</td>
      <td>0.231861</td>
      <td>0.221899</td>
      <td>0.274597</td>
      <td>0.220187</td>
      <td>0.217476</td>
      <td>0.247799</td>
      <td>...</td>
      <td>0.242508</td>
      <td>0.267151</td>
      <td>0.241665</td>
      <td>0.297465</td>
      <td>0.277252</td>
      <td>0.254850</td>
      <td>0.215778</td>
      <td>0.273919</td>
      <td>0.213626</td>
      <td>0.221497</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.254301</td>
      <td>0.276815</td>
      <td>0.262079</td>
      <td>0.276653</td>
      <td>0.223491</td>
      <td>0.245711</td>
      <td>0.245537</td>
      <td>0.263732</td>
      <td>0.262245</td>
      <td>0.231878</td>
      <td>...</td>
      <td>0.204481</td>
      <td>0.209283</td>
      <td>0.240290</td>
      <td>0.233731</td>
      <td>0.200448</td>
      <td>0.287957</td>
      <td>0.271286</td>
      <td>0.224508</td>
      <td>0.227904</td>
      <td>0.236192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.237774</td>
      <td>0.202855</td>
      <td>0.247976</td>
      <td>0.317224</td>
      <td>0.223387</td>
      <td>0.233426</td>
      <td>0.264192</td>
      <td>0.212201</td>
      <td>0.223418</td>
      <td>0.249744</td>
      <td>...</td>
      <td>0.238321</td>
      <td>0.267640</td>
      <td>0.241903</td>
      <td>0.294343</td>
      <td>0.274822</td>
      <td>0.248272</td>
      <td>0.203483</td>
      <td>0.266501</td>
      <td>0.216829</td>
      <td>0.237824</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.256095</td>
      <td>0.280523</td>
      <td>0.260821</td>
      <td>0.280272</td>
      <td>0.235922</td>
      <td>0.238802</td>
      <td>0.235931</td>
      <td>0.267297</td>
      <td>0.253374</td>
      <td>0.229589</td>
      <td>...</td>
      <td>0.211445</td>
      <td>0.206317</td>
      <td>0.235951</td>
      <td>0.228893</td>
      <td>0.201917</td>
      <td>0.283589</td>
      <td>0.277336</td>
      <td>0.223785</td>
      <td>0.231298</td>
      <td>0.222376</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.264856</td>
      <td>0.261657</td>
      <td>0.263420</td>
      <td>0.315670</td>
      <td>0.230888</td>
      <td>0.236793</td>
      <td>0.252360</td>
      <td>0.242478</td>
      <td>0.206340</td>
      <td>0.266352</td>
      <td>...</td>
      <td>0.225202</td>
      <td>0.274650</td>
      <td>0.229674</td>
      <td>0.282011</td>
      <td>0.223278</td>
      <td>0.272895</td>
      <td>0.206819</td>
      <td>0.241059</td>
      <td>0.235670</td>
      <td>0.228535</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <td>0.253545</td>
      <td>0.244569</td>
      <td>0.276877</td>
      <td>0.303954</td>
      <td>0.213353</td>
      <td>0.251170</td>
      <td>0.265480</td>
      <td>0.225302</td>
      <td>0.261119</td>
      <td>0.196203</td>
      <td>...</td>
      <td>0.196982</td>
      <td>0.235612</td>
      <td>0.257314</td>
      <td>0.241572</td>
      <td>0.208169</td>
      <td>0.255695</td>
      <td>0.213576</td>
      <td>0.228342</td>
      <td>0.204193</td>
      <td>0.210340</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>0.241263</td>
      <td>0.232127</td>
      <td>0.228427</td>
      <td>0.295329</td>
      <td>0.252916</td>
      <td>0.222518</td>
      <td>0.239046</td>
      <td>0.260487</td>
      <td>0.239470</td>
      <td>0.278681</td>
      <td>...</td>
      <td>0.238573</td>
      <td>0.222199</td>
      <td>0.220869</td>
      <td>0.283524</td>
      <td>0.259816</td>
      <td>0.275169</td>
      <td>0.256728</td>
      <td>0.264117</td>
      <td>0.223195</td>
      <td>0.244809</td>
    </tr>
    <tr>
      <th>3997</th>
      <td>0.267901</td>
      <td>0.238164</td>
      <td>0.249795</td>
      <td>0.310199</td>
      <td>0.215283</td>
      <td>0.217200</td>
      <td>0.236712</td>
      <td>0.250475</td>
      <td>0.254011</td>
      <td>0.232966</td>
      <td>...</td>
      <td>0.191824</td>
      <td>0.224332</td>
      <td>0.247546</td>
      <td>0.258202</td>
      <td>0.232979</td>
      <td>0.249462</td>
      <td>0.212318</td>
      <td>0.231168</td>
      <td>0.230317</td>
      <td>0.194971</td>
    </tr>
    <tr>
      <th>3998</th>
      <td>0.264819</td>
      <td>0.259637</td>
      <td>0.249645</td>
      <td>0.295401</td>
      <td>0.214651</td>
      <td>0.228113</td>
      <td>0.240940</td>
      <td>0.259314</td>
      <td>0.223928</td>
      <td>0.220803</td>
      <td>...</td>
      <td>0.211240</td>
      <td>0.202161</td>
      <td>0.255878</td>
      <td>0.257834</td>
      <td>0.223692</td>
      <td>0.248534</td>
      <td>0.232569</td>
      <td>0.208684</td>
      <td>0.225532</td>
      <td>0.208274</td>
    </tr>
    <tr>
      <th>3999</th>
      <td>0.255834</td>
      <td>0.272207</td>
      <td>0.247352</td>
      <td>0.298092</td>
      <td>0.216819</td>
      <td>0.218597</td>
      <td>0.266582</td>
      <td>0.230202</td>
      <td>0.252146</td>
      <td>0.245384</td>
      <td>...</td>
      <td>0.217986</td>
      <td>0.197049</td>
      <td>0.245754</td>
      <td>0.261453</td>
      <td>0.218645</td>
      <td>0.276013</td>
      <td>0.235816</td>
      <td>0.201815</td>
      <td>0.198125</td>
      <td>0.217942</td>
    </tr>
  </tbody>
</table>
<p>4000 rows × 656 columns</p>
</div>



For every player, we can use the above set of Stan estimates to get a point estimate of their gamma (we'll use the mean; you can also use the median), and an uncertainty interval that covers 95% of the Stan estimates.


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
      <td>0.244787</td>
      <td>0.218377</td>
      <td>0.272459</td>
    </tr>
    <tr>
      <th>1</th>
      <td>abreujo02</td>
      <td>540</td>
      <td>128</td>
      <td>0.237037</td>
      <td>0.240307</td>
      <td>0.214045</td>
      <td>0.267754</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abreuwi02</td>
      <td>76</td>
      <td>24</td>
      <td>0.315789</td>
      <td>0.255204</td>
      <td>0.218271</td>
      <td>0.293940</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>643</td>
      <td>217</td>
      <td>0.337481</td>
      <td>0.300469</td>
      <td>0.272502</td>
      <td>0.329750</td>
    </tr>
    <tr>
      <th>4</th>
      <td>adamewi01</td>
      <td>553</td>
      <td>120</td>
      <td>0.216998</td>
      <td>0.228697</td>
      <td>0.203143</td>
      <td>0.254521</td>
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
      <td>0.268994</td>
      <td>0.240938</td>
      <td>0.297001</td>
    </tr>
    <tr>
      <th>652</th>
      <td>youngja02</td>
      <td>43</td>
      <td>8</td>
      <td>0.186047</td>
      <td>0.238679</td>
      <td>0.203513</td>
      <td>0.276385</td>
    </tr>
    <tr>
      <th>653</th>
      <td>youngja03</td>
      <td>107</td>
      <td>27</td>
      <td>0.252336</td>
      <td>0.245811</td>
      <td>0.209549</td>
      <td>0.284095</td>
    </tr>
    <tr>
      <th>654</th>
      <td>zavalse01</td>
      <td>175</td>
      <td>30</td>
      <td>0.171429</td>
      <td>0.222789</td>
      <td>0.190221</td>
      <td>0.256177</td>
    </tr>
    <tr>
      <th>655</th>
      <td>zuninmi01</td>
      <td>124</td>
      <td>22</td>
      <td>0.177419</td>
      <td>0.229074</td>
      <td>0.194073</td>
      <td>0.264210</td>
    </tr>
  </tbody>
</table>
<p>656 rows × 7 columns</p>
</div>



Here is the roster of top 10 batters, according to the Stan point estimates. Notice that all the players in this roster have been at bat hundreds of times, so we can consider this top 10 to be more trustworthy. 


```python
stan_top10 = battingf.nlargest(10, 'gamma')
stan_top10.loc[:, ['playerID', 'atbat', 'hits', 'batting_avg', 'gamma', 'g_min', 'g_max']]
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
      <td>0.306983</td>
      <td>0.278533</td>
      <td>0.335950</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>643</td>
      <td>217</td>
      <td>0.337481</td>
      <td>0.300469</td>
      <td>0.272502</td>
      <td>0.329750</td>
    </tr>
    <tr>
      <th>200</th>
      <td>freemfr01</td>
      <td>637</td>
      <td>211</td>
      <td>0.331240</td>
      <td>0.296616</td>
      <td>0.268939</td>
      <td>0.325132</td>
    </tr>
    <tr>
      <th>158</th>
      <td>diazya01</td>
      <td>525</td>
      <td>173</td>
      <td>0.329524</td>
      <td>0.291387</td>
      <td>0.262630</td>
      <td>0.320276</td>
    </tr>
    <tr>
      <th>520</th>
      <td>seageco01</td>
      <td>477</td>
      <td>156</td>
      <td>0.327044</td>
      <td>0.288108</td>
      <td>0.257878</td>
      <td>0.319570</td>
    </tr>
    <tr>
      <th>61</th>
      <td>bettsmo01</td>
      <td>584</td>
      <td>179</td>
      <td>0.306507</td>
      <td>0.280058</td>
      <td>0.252020</td>
      <td>0.308918</td>
    </tr>
    <tr>
      <th>62</th>
      <td>bichebo01</td>
      <td>571</td>
      <td>175</td>
      <td>0.306480</td>
      <td>0.280045</td>
      <td>0.252649</td>
      <td>0.308639</td>
    </tr>
    <tr>
      <th>53</th>
      <td>bellico01</td>
      <td>499</td>
      <td>153</td>
      <td>0.306613</td>
      <td>0.278081</td>
      <td>0.248118</td>
      <td>0.308982</td>
    </tr>
    <tr>
      <th>469</th>
      <td>ramirha02</td>
      <td>400</td>
      <td>125</td>
      <td>0.312500</td>
      <td>0.277054</td>
      <td>0.247250</td>
      <td>0.309745</td>
    </tr>
    <tr>
      <th>410</th>
      <td>naylojo01</td>
      <td>452</td>
      <td>139</td>
      <td>0.307522</td>
      <td>0.276984</td>
      <td>0.247191</td>
      <td>0.307539</td>
    </tr>
  </tbody>
</table>
</div>



Let's plot the top 10 players' gammas, along with their observed batting average and the estimated global mean batting ability.



```python
global_ba = abframe['global_ba'].mean()

(
    ggplot(stan_top10, aes(x = "reorder(playerID, -gamma)")) + 
    geom_errorbar(aes(ymin="g_min", ymax="g_max"),color="#1b9e77" ) +
    geom_point(aes(y = "gamma"), color="#1b9e77") + 
    geom_point(aes(y = "batting_avg"), color="#7570b3") + 
    geom_hline(yintercept = global_ba, linetype="dashed") + 
    theme(axis_text_x=element_text(angle=90, hjust=1)) + 
    labs(x="Player ID in Stan-ranked order") + 
    ggtitle(f"Estimated gamma and 95% uncertainty intervals for top 10 players.\nObserved batting average in purple")
)
```


    
![png](baseball_stats_files/baseball_stats_28_0.png)
    


You might wonder why the observed batting averages are consistently so much higher than the estimated batting abilities. This is because the model is smoothing all the estimates towards the priors. Hence, performance estimates for high performing batters will be biased down, and performance estimates for low performing batters will be biased up. This is not a property unique to Stan; it is a property of the smoothing process.

It is also worth pointing out, however, that the uncertainty intervals are away from the estimated global mean (horizontal dashed line), indicating that the model identifies these players as having above average batting ability. Furthermore, the ranking order of the players according to their gamma estimates is generally consistent with the ranking of their empirical batting averages.

What about the players with very few at-bats? As desired, their gammas are near the estimated global mean of 0.24.


```python
battingf.loc[battingf['atbat']<=5, ['playerID', 'atbat', 'hits', 'batting_avg', 'gamma', 'g_min', 'g_max']]
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
      <td>0.243137</td>
      <td>0.201593</td>
      <td>0.286412</td>
    </tr>
    <tr>
      <th>111</th>
      <td>castidi02</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243584</td>
      <td>0.203323</td>
      <td>0.285488</td>
    </tr>
    <tr>
      <th>124</th>
      <td>colliza01</td>
      <td>4</td>
      <td>2</td>
      <td>0.5</td>
      <td>0.246694</td>
      <td>0.205704</td>
      <td>0.290228</td>
    </tr>
    <tr>
      <th>140</th>
      <td>culbech01</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.246049</td>
      <td>0.206210</td>
      <td>0.288343</td>
    </tr>
    <tr>
      <th>165</th>
      <td>downsje01</td>
      <td>5</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.246178</td>
      <td>0.204776</td>
      <td>0.290680</td>
    </tr>
    <tr>
      <th>205</th>
      <td>fulmemi01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243544</td>
      <td>0.203917</td>
      <td>0.285525</td>
    </tr>
    <tr>
      <th>229</th>
      <td>graytr01</td>
      <td>5</td>
      <td>2</td>
      <td>0.4</td>
      <td>0.246226</td>
      <td>0.206139</td>
      <td>0.287573</td>
    </tr>
    <tr>
      <th>242</th>
      <td>hamilbi02</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243047</td>
      <td>0.204422</td>
      <td>0.283931</td>
    </tr>
    <tr>
      <th>243</th>
      <td>hamilca01</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.241305</td>
      <td>0.201175</td>
      <td>0.284631</td>
    </tr>
    <tr>
      <th>290</th>
      <td>kaiseco01</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.241411</td>
      <td>0.202833</td>
      <td>0.283075</td>
    </tr>
    <tr>
      <th>325</th>
      <td>lopezal03</td>
      <td>2</td>
      <td>1</td>
      <td>0.5</td>
      <td>0.245342</td>
      <td>0.205990</td>
      <td>0.285929</td>
    </tr>
    <tr>
      <th>362</th>
      <td>mccoyma01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243426</td>
      <td>0.201870</td>
      <td>0.286100</td>
    </tr>
    <tr>
      <th>386</th>
      <td>millesh01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243544</td>
      <td>0.203130</td>
      <td>0.286071</td>
    </tr>
    <tr>
      <th>388</th>
      <td>mitchca01</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.242039</td>
      <td>0.201150</td>
      <td>0.283373</td>
    </tr>
    <tr>
      <th>424</th>
      <td>okeych01</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.242842</td>
      <td>0.203207</td>
      <td>0.285058</td>
    </tr>
    <tr>
      <th>482</th>
      <td>reynoma03</td>
      <td>5</td>
      <td>1</td>
      <td>0.2</td>
      <td>0.243695</td>
      <td>0.201459</td>
      <td>0.286321</td>
    </tr>
    <tr>
      <th>514</th>
      <td>sborzjo01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243841</td>
      <td>0.204074</td>
      <td>0.286690</td>
    </tr>
    <tr>
      <th>521</th>
      <td>seaglch01</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.243625</td>
      <td>0.204336</td>
      <td>0.284714</td>
    </tr>
    <tr>
      <th>527</th>
      <td>shewmbr01</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.241831</td>
      <td>0.201425</td>
      <td>0.284156</td>
    </tr>
    <tr>
      <th>529</th>
      <td>sianimi01</td>
      <td>5</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.241543</td>
      <td>0.199343</td>
      <td>0.284477</td>
    </tr>
    <tr>
      <th>614</th>
      <td>vilorme01</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.242775</td>
      <td>0.202628</td>
      <td>0.284301</td>
    </tr>
    <tr>
      <th>622</th>
      <td>wainwad01</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.242915</td>
      <td>0.201881</td>
      <td>0.286345</td>
    </tr>
  </tbody>
</table>
</div>



## Who's the best player? Another way to choose

If our goal is in fact to choose the player(s) with the highest batting ability, there is another way to do it, using the "possible worlds" sampled by Stan. For each one of the 4000 possible worlds, we identify the best performing player. The "true" best player is most likely the one who is best in the most possible worlds.

Below, we find the best player in every Stan sample, and pick our top 10 accordingly. We could of course pick the top 10 in each possible world, and draw our "most likely top 10" from the resulting sets, but picking the single best is easier to code, and gets the point across.


```python
players = batting_estimates.columns

# get the best performance in each sample world
best_perf = batting_estimates.max(axis=1)

# mark which player was the best in each world. ties ok
is_best = batting_estimates.eq(best_perf, axis=0).astype(int) 

mean_best = is_best.mean().reset_index() # compute the series and convert it to a data frame
mean_best.columns = ['playerID', 'frac_as_best']

# join it into battingf
battingf = battingf.merge(mean_best, on='playerID')
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
      <th>gamma</th>
      <th>g_min</th>
      <th>g_max</th>
      <th>frac_as_best</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>abramcj01</td>
      <td>563</td>
      <td>138</td>
      <td>0.245115</td>
      <td>0.244787</td>
      <td>0.218377</td>
      <td>0.272459</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>abreujo02</td>
      <td>540</td>
      <td>128</td>
      <td>0.237037</td>
      <td>0.240307</td>
      <td>0.214045</td>
      <td>0.267754</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abreuwi02</td>
      <td>76</td>
      <td>24</td>
      <td>0.315789</td>
      <td>0.255204</td>
      <td>0.218271</td>
      <td>0.293940</td>
      <td>0.00125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>643</td>
      <td>217</td>
      <td>0.337481</td>
      <td>0.300469</td>
      <td>0.272502</td>
      <td>0.329750</td>
      <td>0.17325</td>
    </tr>
    <tr>
      <th>4</th>
      <td>adamewi01</td>
      <td>553</td>
      <td>120</td>
      <td>0.216998</td>
      <td>0.228697</td>
      <td>0.203143</td>
      <td>0.254521</td>
      <td>0.00000</td>
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
      <td>...</td>
    </tr>
    <tr>
      <th>651</th>
      <td>yoshima02</td>
      <td>537</td>
      <td>155</td>
      <td>0.288641</td>
      <td>0.268994</td>
      <td>0.240938</td>
      <td>0.297001</td>
      <td>0.00200</td>
    </tr>
    <tr>
      <th>652</th>
      <td>youngja02</td>
      <td>43</td>
      <td>8</td>
      <td>0.186047</td>
      <td>0.238679</td>
      <td>0.203513</td>
      <td>0.276385</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>653</th>
      <td>youngja03</td>
      <td>107</td>
      <td>27</td>
      <td>0.252336</td>
      <td>0.245811</td>
      <td>0.209549</td>
      <td>0.284095</td>
      <td>0.00025</td>
    </tr>
    <tr>
      <th>654</th>
      <td>zavalse01</td>
      <td>175</td>
      <td>30</td>
      <td>0.171429</td>
      <td>0.222789</td>
      <td>0.190221</td>
      <td>0.256177</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>655</th>
      <td>zuninmi01</td>
      <td>124</td>
      <td>22</td>
      <td>0.177419</td>
      <td>0.229074</td>
      <td>0.194073</td>
      <td>0.264210</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
<p>656 rows × 8 columns</p>
</div>




```python
# top 10 by fraction best
top10_by_frac = battingf.nlargest(10, 'frac_as_best')
top10_by_frac[['playerID', 'frac_as_best', 'batting_avg', 'gamma']]
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
      <th>frac_as_best</th>
      <th>batting_avg</th>
      <th>gamma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>arraelu01</td>
      <td>0.33725</td>
      <td>0.353659</td>
      <td>0.306983</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>0.17325</td>
      <td>0.337481</td>
      <td>0.300469</td>
    </tr>
    <tr>
      <th>200</th>
      <td>freemfr01</td>
      <td>0.11475</td>
      <td>0.331240</td>
      <td>0.296616</td>
    </tr>
    <tr>
      <th>158</th>
      <td>diazya01</td>
      <td>0.05650</td>
      <td>0.329524</td>
      <td>0.291387</td>
    </tr>
    <tr>
      <th>520</th>
      <td>seageco01</td>
      <td>0.04650</td>
      <td>0.327044</td>
      <td>0.288108</td>
    </tr>
    <tr>
      <th>53</th>
      <td>bellico01</td>
      <td>0.01350</td>
      <td>0.306613</td>
      <td>0.278081</td>
    </tr>
    <tr>
      <th>469</th>
      <td>ramirha02</td>
      <td>0.01275</td>
      <td>0.312500</td>
      <td>0.277054</td>
    </tr>
    <tr>
      <th>62</th>
      <td>bichebo01</td>
      <td>0.01125</td>
      <td>0.306480</td>
      <td>0.280045</td>
    </tr>
    <tr>
      <th>61</th>
      <td>bettsmo01</td>
      <td>0.01100</td>
      <td>0.306507</td>
      <td>0.280058</td>
    </tr>
    <tr>
      <th>18</th>
      <td>altuvjo01</td>
      <td>0.00850</td>
      <td>0.311111</td>
      <td>0.274941</td>
    </tr>
  </tbody>
</table>
</div>



This is substantially the same set of players as were selected by looking just at the point estimates. That's good! It gives us confidence that these are indeed the players with the highest batting ability.

Let's mark the players who show up in the top 10, by either criterion.


```python
# top 10 by point estimate of performance -- the columns are ordered differently

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

    Picked by point estimate but not by fraction best: {'naylojo01'}
    Picked by fraction best but not by point estimate: {'altuvjo01'}





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
      <td>0.300469</td>
      <td>0.17325</td>
    </tr>
    <tr>
      <th>18</th>
      <td>altuvjo01</td>
      <td>360</td>
      <td>112</td>
      <td>0.311111</td>
      <td>0.274941</td>
      <td>0.00850</td>
    </tr>
    <tr>
      <th>34</th>
      <td>arraelu01</td>
      <td>574</td>
      <td>203</td>
      <td>0.353659</td>
      <td>0.306983</td>
      <td>0.33725</td>
    </tr>
    <tr>
      <th>53</th>
      <td>bellico01</td>
      <td>499</td>
      <td>153</td>
      <td>0.306613</td>
      <td>0.278081</td>
      <td>0.01350</td>
    </tr>
    <tr>
      <th>61</th>
      <td>bettsmo01</td>
      <td>584</td>
      <td>179</td>
      <td>0.306507</td>
      <td>0.280058</td>
      <td>0.01100</td>
    </tr>
    <tr>
      <th>62</th>
      <td>bichebo01</td>
      <td>571</td>
      <td>175</td>
      <td>0.306480</td>
      <td>0.280045</td>
      <td>0.01125</td>
    </tr>
    <tr>
      <th>158</th>
      <td>diazya01</td>
      <td>525</td>
      <td>173</td>
      <td>0.329524</td>
      <td>0.291387</td>
      <td>0.05650</td>
    </tr>
    <tr>
      <th>200</th>
      <td>freemfr01</td>
      <td>637</td>
      <td>211</td>
      <td>0.331240</td>
      <td>0.296616</td>
      <td>0.11475</td>
    </tr>
    <tr>
      <th>410</th>
      <td>naylojo01</td>
      <td>452</td>
      <td>139</td>
      <td>0.307522</td>
      <td>0.276984</td>
      <td>0.00775</td>
    </tr>
    <tr>
      <th>469</th>
      <td>ramirha02</td>
      <td>400</td>
      <td>125</td>
      <td>0.312500</td>
      <td>0.277054</td>
      <td>0.01275</td>
    </tr>
    <tr>
      <th>520</th>
      <td>seageco01</td>
      <td>477</td>
      <td>156</td>
      <td>0.327044</td>
      <td>0.288108</td>
      <td>0.04650</td>
    </tr>
  </tbody>
</table>
</div>



Now let's plot all the players, sorted by observed batting average (lowest to highest). We'll plot the estimated player ability (`gamma`), along with the 95% uncertainty intervals around the estimates (in light gray).
The points are also color coded by whether or not the player made at least one of the top 10 lists (in green) or not (in purple). The dashed line is the estimated mean player ability.


```python

palette = {
    'False':'#7570b3',
    'True': '#1b9e77'
}

labels = {
    'False': 'Not in top 10',
    'True': 'In top 10 by some criterion'
}

(
    ggplot(battingf, aes(x = "reorder(playerID, batting_avg)")) + 
    geom_errorbar(aes(ymin="g_min", ymax="g_max"), color='lightgray' ) +
    geom_point(aes(y = "gamma", color='in_top10')) + 
    geom_hline(yintercept = global_ba, linetype="dashed") + 
    labs(x="Players ordered by observed batting average (lowest to highest)", y="player ability (baselined at estimated mean ability)") + 
    ggtitle(f"Estimated player ability with 95% uncertainty interval, compared to observed performance") + 
    scale_color_manual(values=palette, labels=labels) +
    theme(figure_size = (12, 8), legend_position='bottom', axis_text_x=element_blank(), axis_ticks_x=element_blank())
)


```


    
![png](baseball_stats_files/baseball_stats_38_0.png)
    


There are a few things to note in this graph. First, the ranking of ability estimates roughly correlate with the ranking of observed batting averages, as desired. There is a cluster of purple players to the right of the green players who have relatively low (actually, average) estimated batting ability, but high observed batting average. These are the players who weren't at bat very often, but who were successful when they were. The model did not have enough data on these players to move the ability estimates away from the prior. Similarly, there are players at the far left of the graph who cluster around the mean, even though their observed batting averages are zero, or nearly so. These are also players with only a few at bats, so their estimated abilities still smooth strongly into the prior.

So if the goal of estimating player ability is to identify the best players, then this model has been able to do so. It automatically discounts spurious empirical estimates that are likely inaccurate due to insufficient data, without the analyst having to specify what "insufficient data" is in an ad-hoc way. It also provides reasonable assumptions about the abilities of low information (low at-bat) players---assumptions that are based on the population data.

## Matching the model to reality


```python
battingf.nlargest(3, 'frac_as_best')[['playerID', 'frac_as_best', 'batting_avg', 'gamma']]
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
      <th>frac_as_best</th>
      <th>batting_avg</th>
      <th>gamma</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>arraelu01</td>
      <td>0.33725</td>
      <td>0.353659</td>
      <td>0.306983</td>
    </tr>
    <tr>
      <th>3</th>
      <td>acunaro01</td>
      <td>0.17325</td>
      <td>0.337481</td>
      <td>0.300469</td>
    </tr>
    <tr>
      <th>200</th>
      <td>freemfr01</td>
      <td>0.11475</td>
      <td>0.331240</td>
      <td>0.296616</td>
    </tr>
  </tbody>
</table>
</div>



Stan's selected best-ranked player by both "probability of being best" and point estimate criteria, `arraelu01`, is player Luis Arráez. Here's what his [Wikipedia page](https://en.wikipedia.org/wiki/Luis_Arr%C3%A1ez) has to say about him:

> Known for his ability to put the ball in play and not striking out, Arráez is considered one of the best contact hitters of his generation. From 2022 to 2024, Arráez became the first player in MLB history to win three consecutive batting titles with three different teams.... He was also the second player in the modern era to win a batting title in each league and the first to do so in consecutive years. 

Note that his career MLB batting average[^2] (calculated from 2019 through May 17, 2026) is 0.317. This is pretty close to our estimated `gamma` of 0.307; in fact, our estimate is a better prediction of Arráez's career performance (so far) than the simple observation of his 2023 season batting average is.

The next two players (as ranked by probability of being best) are Ronald Acuña, Jr. (career batting average so far = 0.288) and Freddie Freeman (career batting average so far = 0.299). Again, both Stan's estimates and career batting averages for all these players are below their observed 2023 season batting averages---showing that smoothing performance estimates to the population mean was a reasonable modeling choice. Note also that the career batting averages for these top three players also fell within the 95% uncertainty intervals estimated by Stan. 


[^2]: All career batting averages as given by Wikipedia on May 19, 2026.

## Estimate what you want to know, not just what you can observe

As we've seen in the above example, an advantage of probabilistic modeling is that the analyst is able to distinguish between *observations* and (potentially unobservable) *quantities of interest*. If you, the analyst, can describe a probabilistic process that relates *what you can see* to *what you actually need to know*, then probabilistic modeling programs like Stan can estimate these quantities for you. By specifying the process to describe your problem and your task goal, you can add in prior knowledge or assumptions about the domain in a principled, documentable way, without having to resort to ad-hoc tweaks or data processing. 

In addition, probabilistic modeling systems that are based on Monte Carlo sampling (like Stan) provide samples of "possible worlds" that are consistent with the training data. You can use these samples not only to calculate point estimates of quantities of interest, but also uncertainty intervals around those estimates. You can also use the possible worlds to run simulations and scenarios (like, "who are the top 10 players in each possible world?") to further help you in decision-making. 

Of course, the estimates can only be as good as the process that you describe. But as your understanding of and intuitions about these processes improve, then so, too,  can your model. 
