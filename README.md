# MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Manalo, Zuriel | 21-02255 | MEXE 4102 |
| Mercado, Rica Jean | 24-04909 | MEXE 4102 |

## Notebook links

| Chapter | Links |
|---|---|
| Ch1_2_3 | [Introduction to Data Pre-processing, The Power of Data: Initial Steps in Loading, Understanding, and Exploring Data with Python, Cleaning Your Data](https://colab.research.google.com/drive/1tjeuCZW5-EVsePXqjGKmnh8FQzo0NsrB?usp=sharing) |
| Ch4 | [Unleashing the Power of Data Through Transformation and Feature Engineering](https://colab.research.google.com/drive/1RjX8WbJ-GnfDiDBJtoFO66FH5cHFuFfc?usp=sharing) |
| Ch5 | [Unfolding the Essentials of Data Scaling and Normalization](https://colab.research.google.com/drive/10GnAQJPKp6tjJSnQN1HprRnWvyEAkuKK?usp=sharing) |
| Ch6 | [Outlier detection](https://colab.research.google.com/drive/1e6V4F0GQJkxs8rxUVWyjBOzuQQ-VUPT0#scrollTo=xw-ulSLhnqbd) |
| Ch7 | [Feature selection](https://colab.research.google.com/drive/1-okiKPYDlmhkfBnKtyHIJr5GcAOlnGCf#scrollTo=oIdgmuT_PTv6) |
| Ch8 | [Constructing a preprocessing pipeline](https://colab.research.google.com/drive/12G8S06ZgwGvD_NCahoSLdwqvkBfBb921#scrollTo=y7rpcwtM1ALR) |
| Ch9 | [Full pipeline and visualization](https://colab.research.google.com/drive/16w7chxNDIuME0EoYpJATHmPmOhYGkieq#scrollTo=Ikv0W_I7hvhA) |

# What We Learned

### Chapters 1–3: Data Preprocessing

In Chapter 1, 2, and 3, we learned what data preprocessing is and why it is essential for machine learning models. In Chapter 3, we were specifically surprised since we didn't know that there was a way to find blank or missing values using a line of code.

### Chapter 4: Feature Engineering

Chapter 4 taught us how to turn raw data into useful features. We were able to find the relationships of variables using interaction features. We also learned about binning, which groups and labels datasets. Lastly, we learned the two types of encoding, which are one-hot and ordinal, and when to use each categorical encoding that would fit for the dataset.

### Chapter 5: Feature Scaling

In Chapter 5, we learned how to standardize the dataset so that it would be easier to see the difference between individual data. Additionally, we found out what scaling was and why it was needed. It taught us that without it, the model would prioritize the larger set numbers and ignore the smaller set of numbers.

### Chapter 6: Dealing with Outliers

This chapter taught how outliers skew datasets and distort overall averages, functioning like a noticeable disruptive entity within normal statistical trends. I learned that both the Z-score and Interquartile Range (IQR) methods can programmatically isolate anomalous variations, which is critical for making analysis accurate and models highly reliable. What surprised me was that an isolated mathematical outlier under one standard definition (like an IQR fence) might pass undetected under a conservative threshold of another (like a strict Z-score cutoff of 3).

### Chapter 7: Feature Selection

This chapter taught how to eliminate irrelevant features to simplify model training, optimize computational speeds, and prevent noisy parameters from degrading overall predictive accuracy. I understood the structural and procedural divisions separating filter methods (statistical scoring metrics), wrapper methods (treating selections as an iterative model search), and embedded methods (adjusting weights directly during model compilation). What surprised me was how aggressive wrapper algorithms like RFECV are, stripping the student feature pool down to a solitary parameter while regularization methods like LassoCV retained multiple distinct inputs.

### Chapter 8: Constructing a Preprocessing Pipeline

This chapter taught how to automate a structured series of sequential cleaning, imputation, and transformation blocks into an orderly, single-object workflow. By visualizing the pipeline structure as an automated manufacturing conveyor belt, I understood how complex arrays of raw data enter raw at one terminal and exit properly configured and normalized for ML use. What surprised me was learning how `ColumnTransformer` lets you isolate and target distinct pipelines to specific subsets of data columns while entirely avoiding leakage or manually mutating the underlying data structure.

### Chapter 9: Real-World Application: Data Preprocessing

This chapter taught how to combine different numerical and categorical data types from a real-world repository into a unified, consolidated preprocessing schema. I learned how to discretize continuous values into isolated categorical bins to improve classification and how to leverage post-processing heatmaps and plots to confirm that data quality remains robust. What surprised me was discovering how iterative data preparation truly is; tweaking bin sizes or changing imputation strategies drastically alters feature balances, proving preprocessing requires a lot of experimentation.

---

# Errors We Found

### Chapter 4: Temperature Category

While testing Chapter 4, we found that the Temperature Category was displaying the wrong values. The Temperature was 75, but the Temperature Category indicated it was Cool even if, according to the bin and labels, it should be warm.

We added `right=False` to the code, making it:

```python
df['Temperature Category'] = pd.cut(
    df['Temperature'],
    bins=bins,
    labels=labels,
    right=False
)
```

After that, the Temperature Category displayed the correct label for the given temperature.

### Chapter 6: Z-Score Sample Correction

In the original file, `scipy.stats.zscore(data)` is calculated on an array of length 8. The text evaluates it under standard sample definition assumptions, but `scipy.stats.zscore` defaults to a population metric (`ddof=0`). To calculate the true sample-corrected deviation score, it should be updated to `stats.zscore(data, ddof=1)`.

**Correction:**

```python
stats.zscore(data, ddof=1)
```

### Chapter 6: IQR Interpolation Method

The original notebook computes percentiles using `data.quantile(0.25)` and `data.quantile(0.75)`, yielding an IQR of `9.25`. This conflicts with the hand-calculated textbook definition of finding exact split medians (which yields a true IQR of `9.5`). The error should be corrected by setting the parameter to `interpolation='midpoint'` to match custom manual statistical splits.

**Correction:**

```python
data.quantile(0.25, interpolation='midpoint')
data.quantile(0.75, interpolation='midpoint')
```

### Chapter 9: Inverted Discretization Plot Logic

In the provided notebook, the binning line `data['Age'] = pd.cut(data['Age'], bins=bins, labels=labels)` is run inside Cell 13. Immediately following in Cell 15, the notebook attempts to output a numeric distribution using `plt.hist(data['Age'].dropna())` while labeling the output string as `# Before discretization`.

This creates an execution error because the original array column was already mutated into qualitative text categories, preventing a clean baseline histogram from rendering correctly. The pipeline code should be modified to preserve the original column or generate a distinct feature mapping array (`data['Age_binned']`).

**Suggested correction:**

```python
data['Age_binned'] = pd.cut(
    data['Age'],
    bins=bins,
    labels=labels
)
```

---

# Note on AI Tools

Yes, we used AI tools such as Gemini and ChatGPT during the development and review of our notebooks.

###  Gemini

We used Gemini recommendations while coding when we encountered errors in our code. It helped us identify where the error was located, understand what caused it, and determine what we could do to fix it.

###  ChatGPT

We used ChatGPT to help clarify theoretical data concepts, formulate the structural outlines for our markdown summaries, and systematically organize our final report responses.

Using these tools allowed us to better understand the mathematical constraints behind machine learning pipelines and cross-validation folds when working with limited datasets.

---

# References

* McKinney, W. (2021). *Python for Data Analysis* (3rd ed.). O'Reilly.
* VanderPlas, J. *Python Data Science Handbook*.



