import pandas as pd
import numpy as np
import time
import argparse
from scipy import stats as sps
from statistics import mean


parser = argparse.ArgumentParser(description='Calcuate Rt for Covid-19 -> INDIA')
parser.add_argument('-s', '--state', help='State name')
args = parser.parse_args()
states = args.state



def covid_data(url):
    # Calling the covid19india.org's API and parsing CSV file to a dataframe
    df = pd.read_csv(url,
                    usecols=['Updated On', 'State', 'Positive', 'Total Tested'],
                    index_col=['State', 'Updated On'],
                    squeeze=True)
    
    # Drop rows which doesn't have all the values
    df = df.dropna()
    
    # Rename "Positive" to "cases" and "Total Tested" to "tests"
    df = df.rename(columns={'Positive': 'cases', 'Total Tested': 'tests'})            

    return df



def prepare_cases(cases, cutoff_cases=5, cutoff_tests=10):
    #new_cases = cases.diff()

    smoothed = cases.rolling(7,
                    win_type='gaussian',
                    min_periods=1,
                    center=True).mean(std=2).round()

    idx_start_cases = np.searchsorted(smoothed['cases'], cutoff_cases)
    idx_start_tests = np.searchsorted(smoothed['tests'], cutoff_tests)
    idx_start = max(idx_start_tests, idx_start_cases)
    smoothed = smoothed.iloc[idx_start:]
    original = cases.loc[smoothed.index]
    
    return original, smoothed



def get_posteriors(sr, sigma=0.15):
    
    # (1) Calculate p
    odds = sr['positivityOdds'].values[:-1] * np.exp(GAMMA * (r_t_range[:, None] - 1)) # / sr['tests'].values[:-1]
    p = odds / (1+odds)

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
                    data = sps.binom.pmf(sr['cases'].values[1:], p=p, n=sr['tests'].values[1:]),
                    index = r_t_range,
                    columns = sr.index[1:]
                  )
    likelihoods[likelihoods.isna()] = 0

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range, scale=sigma).pdf(r_t_range[:, None]) 

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    #prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range)/len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
                    index=r_t_range,
                    columns=sr.index,
                    data={sr.index[0]: prior0}
                  )
    
    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):

        #(5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        #(5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior
        
        #(5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)
        
        # Execute full Bayes' Rule
        posteriors[current_day] = numerator/denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)
    
    return posteriors, log_likelihood




def highest_density_interval(pmf, p=.9):
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf], index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    
    # Find the smallest range (highest density)
    if len(lows)>0:
        best = (highs - lows).argmin()
        low = pmf.index[lows[best]]
        high = pmf.index[highs[best]]
        return pd.Series([low, high], index=[f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])
    else:
        return pd.Series([0, 0], index=[f'Low_{p*100:.0f}', f'High_{p*100:.0f}'])



start = time.time()

url = 'https://api.covid19india.org/csv/latest/statewise_tested_numbers_data.csv'
GAMMA = 1/7
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)


df = covid_data(url)


# Choose optimal sigma
sigmas = np.linspace(1/20, 1, 20)
if type(states) != list :
    states = [states]
else:
    states = [item for item in states]

df = df.loc[states]

results = {}



print("\nSmoothing the cases...")
for state_name, cases in df.groupby(level='State'):
    
    print(state_name)
    cases = df.loc[state_name].cases
    if sum(cases) == 0:
        print("0 cases reported for " + state_name + " :)")
        continue
    elif sum(cases)<100:
        print("Cases < 100 for " + state_name + " - too low!!!")
        continue
    
    new_cases = df.loc[state_name]
    original, smoothed = prepare_cases(new_cases)
    
    original['positivityOdds']  = original['cases'] /(original['tests'] - original['cases'])
    smoothed['positivityOdds'] = smoothed['cases'] /(smoothed['tests'] - smoothed['cases'])
    
    result = {}
    
    # Holds all posteriors with every given value of sigma
    result['posteriors'] = []
    
    # Holds the log likelihood across all k for each value of sigma
    result['log_likelihoods'] = []
    
    for sigma in sigmas:
        try:
            posteriors, log_likelihood = get_posteriors(smoothed, sigma=sigma)
        except:
            print("Too low cases for " + state_name + "-> sigma: " + str(round(sigma, 2)))
        result['posteriors'].append(posteriors)
        result['log_likelihoods'].append(log_likelihood)
    
    # Store all results keyed off of state name
    results[state_name] = result

print('Done.')



# Each index of this array holds the total of the log likelihoods for the corresponding index of the sigmas array.
total_log_likelihoods = np.zeros_like(sigmas)

# Loop through each state's results and add the log likelihoods to the running total.
for state_name, result in results.items():
    total_log_likelihoods += result['log_likelihoods']

# Select the index with the largest log likelihood total
max_likelihood_index = total_log_likelihoods.argmax()

final_results = None



print("\nComputing final results...")
for state_name, result in results.items():
    print(state_name)
    posteriors = result['posteriors'][max_likelihood_index]
    hdis_90 = highest_density_interval(posteriors, p=.9)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis_90], axis=1)
    result["State"] = state_name
    if final_results is None:
        final_results = result
    else:
        final_results = pd.concat([final_results, result])

# Since we now use a uniform prior, the first datapoint is pretty bogus, so just truncating it here
final_results = final_results.groupby('State').apply(lambda x: x.iloc[1:])
# Groupby leaves a duplicate "State" column but as it is already in index so, removing it here
final_results = final_results.drop(columns="State")

# Creating a dataframe to merge no. of cases per day in original dataframe
temp = pd.DataFrame()
for state_name in states:
    df_temp = pd.DataFrame(df.loc[state_name].cases)
    df_temp["State"] = state_name
    temp = temp.append(df_temp.rename(columns={state_name:"Cases"}))
temp = temp.reset_index().set_index(["State", "Updated On"])

# Merging
final_results = final_results.merge(temp, on= ["State", "Updated On"])
#Renaming a few columns
final_results = final_results.rename(columns={"ML": "Rt", "Low_90": "Low", "High_90": "High"})
#Resetting an index
final_results = final_results.reset_index(level=['Updated On'], inplace=False)
print("Done!!!")
print("\nIt took " + str(round((time.time()-start), 3)) + " secs.!")

final_results.to_csv("rt_india.csv")
print("\nRt data exported to CSV -> rt_india.csv")