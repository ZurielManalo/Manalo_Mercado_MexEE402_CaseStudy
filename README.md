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
| Ch6 | [link]() |
| Ch7 | [link]() |
| Ch8 | [link]() |
| Ch9 | [link]() |

## What we learned

&emsp;In Chapter 1 2 and 3 we learned what data preprocessing is and why it is essential for machine learning models. In chapter 3 we were specifically surprised since we didn't know that there was a way to find blank or missing values using a line of code.\
&emsp; Chapter 4 taught us how to turn raw data into useful features, we were able to find the relationships of variables using interaction features. We also learned about binning which groups and labels datasets. Lastly the two types of encoding which are one-hot and ordinal and when to use each categorical encoding that would fit for the dataset.\
&emsp; In Chapter 5 we learned how to standardize the data set so that it would be easier to see the difference between individual data. Additionally, we found out what scaling was and why it was needed. It taught us that without it the model would prioritize the larger set numbers and ignore the smaller set of numbers. 
## Errors we found
&emsp;While testing Chapter 4 i found that the Temperature Category was displaying the wrong values, the Temperature was 75 but the Temperature Category indicated it was Cool even if according to the bin and labels it should be warm. I added `right=False` to the code making it `df['Temperature Category'] = pd.cut(df['Temperature'], bins=bins, labels=labels, right=False )`. After that the Temperature Category displayed the correct label for the given temperature\

## Note on AI tools
Google Gemini for definitions, explanations, and analysis of code.

## References

