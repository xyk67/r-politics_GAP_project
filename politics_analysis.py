import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from scipy.stats import levene

politics = pd.read_csv('rpolitics_engagement.csv')

#Data cleaning

#Converter to fix instances of comma in values and convert 'M' to million in values
def converter(value):
    value = str(value).strip().replace(',', '')
    if value.endswith('M'):
        return float(value[:-1]) * 1000_000

    return float(value)

politics['Country'] = politics['Country'].str.strip()
politics["Engagement"] = politics['Upvotes'] + politics['Comments']  #Can always apply multiplier effect to give comments more weight
politics["GDP per capita"] = politics['GDP per capita'].apply(converter)
politics["Population"] = politics["Population"].apply(converter)
politics = politics[politics["Country"] != "(Nambia)"].reset_index(drop=True) # 'Nambia' (Trump's mispronunciation case), can toggle to remove for analysis although it's only 5 posts

#Aggregate table
relevant = politics[["Country",'GDP per capita', "Population","Engagement"]]

summary = relevant.groupby("Country").agg(
    Mean_Engagement=("Engagement", "mean"),
    Median_Engagement=("Engagement", "median"),
    Std_Engagement=("Engagement", "std"),
    Total_country_Engagement=("Engagement", "sum"),
    Engagement_Count=("Engagement", "count"),
    Avg_GDP_per_capita=("GDP per capita","mean"),
    Avg_Population=("Population","mean")
).reset_index().round(2)

summary= summary.sort_values("Mean_Engagement", ascending=False).reset_index(drop=True)

display_table = summary.copy()
display_table["Mean_Engagement"]=display_table["Mean_Engagement"].apply(lambda x: f"{x/1000:.1f}K" if x >1000 else f"{x:.1f}") # Converts big values to shorthand
display_table["Avg_Population"]=display_table["Avg_Population"].apply(lambda x: f"{x/1000_000:.1f}M" if x> 1000_000 else f"{x:.1f}")
print(display_table.to_string(index=False))

#Apply logarithmic for engagement (homoscedasticity violated)

clean_data = summary.dropna()
clean_data ["Log_Engagement"] = np.log(clean_data["Mean_Engagement"])

#Multiple linear regression (GAP model)
#Mean engagement vs Log Engagement to show assumptions violated for 'Mean'

gapmodel = sm.add_constant(clean_data[["Avg_GDP_per_capita","Avg_Population"]])
#Using Mean Engagement as outcome
model_mean = sm.OLS(clean_data['Mean_Engagement'],gapmodel).fit()
print("\nMultiple Linear Regression (Mean engagement): GDP per capita + Population")
print(model_mean.summary())
#Using Log Engagement as outcome
model_log = sm.OLS(clean_data["Log_Engagement"], gapmodel).fit()
print("\nMultiple Linear Regression (Logarithmic engagement): GDP per capita + Population")
print(model_log.summary())

##Assumption checks
#Compare between using 'mean' and 'logarithmic' engagement

residuals = model_mean.resid
fitted = model_mean.fittedvalues
residuals_2 = model_log.resid
fitted_2 = model_log.fittedvalues

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#Homoscedasticity (mean)
axes[0, 0].scatter(fitted, residuals, alpha=0.6, color="steelblue")
axes[0, 0].axhline(0, color="red", linewidth=1.5, linestyle="--")
axes[0, 0].set_title("Residuals vs Fitted (Mean Engagement)")
axes[0, 0].set_xlabel("Fitted Values")
axes[0, 0].set_ylabel("Residuals")

#QQ plot (mean)
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
axes[0, 1].scatter(osm, osr, alpha=0.6, color="steelblue")
axes[0, 1].plot(osm, slope * np.array(osm) + intercept, color="red", linewidth=1.5)
axes[0, 1].set_title("Q-Q Plot (Mean Engagement)")
axes[0, 1].set_xlabel("Theoretical Quantiles")
axes[0, 1].set_ylabel("Sample Quantiles")

#Normality (mean)
axes[0, 2].hist(residuals, bins=10, color="steelblue", edgecolor="white")
axes[0, 2].set_title("Histogram of Residuals (Mean Engagement)")
axes[0, 2].set_xlabel("Residual")
axes[0, 2].set_ylabel("Frequency")

## Log engagement

#Homoscedasticity (log)
axes[1, 0].scatter(fitted_2, residuals_2, alpha=0.6, color="coral")
axes[1, 0].axhline(0, color="red", linewidth=1.5, linestyle="--")
axes[1, 0].set_title("Residuals vs Fitted (Log Engagement)")
axes[1, 0].set_xlabel("Fitted Values")
axes[1, 0].set_ylabel("Residuals")

#QQ plot (log)
(osm2, osr2), (slope2, intercept2, r2) = stats.probplot(residuals_2, dist="norm")
axes[1, 1].scatter(osm2, osr2, alpha=0.6, color="coral")
axes[1, 1].plot(osm2, slope2 * np.array(osm2) + intercept2, color="red", linewidth=1.5)
axes[1, 1].set_title("Q-Q Plot (Log Engagement)")
axes[1, 1].set_xlabel("Theoretical Quantiles")
axes[1, 1].set_ylabel("Sample Quantiles")

#Normality (log)
axes[1, 2].hist(residuals_2, bins=10, color="coral", edgecolor="white")
axes[1, 2].set_title("Histogram of Residuals (log Engagement)")
axes[1, 2].set_xlabel("Residual")
axes[1, 2].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

#Normality shapiro-wilks test (mean)
sw_stat_mean, sw_p_mean = shapiro(residuals)
print(f"Shapiro-Wilk: W = {sw_stat_mean:.3f}, p = {sw_p_mean:.3f}")
print(f"Residuals are {'normally distributed' if sw_p_mean > 0.05 else 'NOT normally distributed'}\n")

#Normality shapiro-wilks test (log)
sw_stat_log, sw_p_log = shapiro(residuals_2)
print(f"Shapiro-Wilk: W = {sw_stat_log:.3f}, p = {sw_p_log:.3f}")
print(f"Residuals are {'normally distributed' if sw_p_log > 0.05 else 'NOT normally distributed'}\n")

#Multicollinearity test (VIF) between GDP per cpaita and population
vif_test = clean_data[["Avg_GDP_per_capita","Avg_Population"]]
vif_data = pd.DataFrame()
vif_data["Variable"] = vif_test.columns
vif_data["VIF"] = [variance_inflation_factor(vif_test.values, i) for i in range(vif_test.shape[1])]
print("VIF (Multicollinearity):")
print(vif_data.to_string(index=False))
print(f"{'No multicollinearity concern' if all(vif_data['VIF'] < 5) else 'Multicollinearity detected'}")


##Trump mention effect analysis

politics["Mention"] = politics["Mention of 'Trump' (No = 0, Yes = 1)"].map({0: "No", 1:"Yes"}) #Trump mention was already dummy coded, this maps 0 and 1 to No and Yes respectively
politics["Mention_dummy"] = politics["Mention of 'Trump' (No = 0, Yes = 1)"] #This makes name shorter

trump = politics[['Country','Mention','Mention_dummy']]
summary_2 = trump.groupby('Country').agg(
    Mention_of_Trump=("Mention", lambda x: f"{(x == 'Yes').sum()} Yes/ {(x == 'No').sum()} No"), #shows count of Trump mention
    Avg_Mention=("Mention_dummy","mean") #shows average mention of Trump over 10 posts for each country, 1.0 = 100%,
).reset_index().round(2)

display_table_2 = summary_2.copy()
print(display_table_2.to_string(index=False))

#T-test for Trump mention vs Non Trump mention on engagement
yes = politics[politics["Mention_dummy"] == 1]["Engagement"]
no  = politics[politics["Mention_dummy"] == 0]["Engagement"]

#Levene's test violated, so Welch's test used
lev_stat, lev_p = levene(yes,no)
lev_display = "<0.001" if lev_p <0.001 else f"{lev_p:.3f}"
print(f"\nLevene's Test: F = {lev_stat:.3f},p = {lev_display}")

equal_var = lev_p > 0.05

results = ttest_ind(yes,no,equal_var=False) #equal variance set to false because Levene's test significant
t_stat = results.statistic
t_p = results.pvalue
df_welch = results.df
p_display = "< 0.001" if t_p < 0.001 else f"= {t_p:.3f}"
print(f"\nT-test: t = {t_stat:.3f}, p {p_display}")
print(f"Mean engagement (Trump mentioned):     {yes.mean():.0f}")
print(f"Mean engagement (Trump not mentioned): {no.mean():.0f}")
print(f"Posts mentioning Trump:     {len(yes)}")
print(f"Posts not mentioning Trump: {len(no)}")
print(f"SD (Trump mentioned):   {yes.std():.0f}")
print(f"SD (Trump not mentioned):   {no.std():.0f}")
print(f"Degrees of freedom: {df_welch:.2f}")

#Bar graph for t-test
fig, axes = plt.subplots(figsize=(7, 10))

means = [no.mean(), yes.mean()]
sds = [no.std(), yes.std()]
labels_bar = ["No Trump Mention", "Trump Mentioned"]
colors_bar = ["steelblue", "coral"]

bars = axes.bar(labels_bar, means, yerr=sds, capsize=8,
                    color=colors_bar, alpha=0.7, edgecolor="white",
                    error_kw=dict(elinewidth=1.5, ecolor="black"))
axes.set_title("Mean Engagement ± SD\nby Trump Mention")
axes.set_ylabel("Mean Engagement")
plt.suptitle("Trump Mention vs Engagement — Welch's Independent Samples T-Test",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("trump_ttest.png", dpi=150, bbox_inches='tight')
plt.show()
