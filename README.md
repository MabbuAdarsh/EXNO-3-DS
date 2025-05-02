## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/376cf646-74bc-418a-9486-3be9f4d3cdd6)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/2140b498-c583-4a5b-a4e9-cd8decd94ca9)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/9ab78ab6-d958-4f30-a2d8-86f82393882a)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/a76edd31-88c4-4b09-9691-b7ab2ee5ab8f)


```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/270192d0-eedd-469e-80f4-ef31ccfc8cb8)

```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/user-attachments/assets/b079fd2b-6b8d-47fe-88c5-e54babc83f7a)

```
pip install --upgrade category_encoders
```

![image](https://github.com/user-attachments/assets/cb2b2632-0463-437b-a19c-0b86d78ea98a)

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```

![image](https://github.com/user-attachments/assets/660d6ec8-8cf6-4b83-abce-cf604175121e)

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```


```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/user-attachments/assets/83c1635a-6268-4e1b-a4ca-f83c4186f734)

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```

![image](https://github.com/user-attachments/assets/6b7c1dfb-a9fb-426c-b3c0-27f5802c576b)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/7d7a7bd8-9be0-4163-9e2f-ab554b9f85c5)

```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/cabc5396-0bd9-48d4-a495-f3570708a0eb)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/f9f9104b-650c-4071-b190-7a61ffa809a1)

```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/98711c02-e5b2-4a1d-a87a-9c018d3950e8)

```
np.square(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/ea3b2833-4ea4-458f-9668-0dd5004e9619)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/9da4c81e-8ab1-40eb-9f51-76841e3f6cec)

```
df.skew()
```

![image](https://github.com/user-attachments/assets/8641eaea-0c75-47be-ba0f-25d7a375934d)

```
df["Highly Negative Skew_yeojohnson"], parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/50415689-5c83-4870-95ba-42710cb2ba7c)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![image](https://github.com/user-attachments/assets/6c738518-fdd7-432c-9774-b584c26a8bc1)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/ca30cc17-0a63-42a7-9c49-3cbe0d001352)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/2212ef7a-535d-46c5-86ee-25279cc71ccf)

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/2a97f132-d852-4082-b5b2-32bab710864f)

```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/5780ee15-ae48-44d8-b8f2-3750e16ff603)

# RESULT:
   Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
