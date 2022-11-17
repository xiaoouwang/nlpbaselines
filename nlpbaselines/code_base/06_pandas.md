# Pandas codebase

## global settings

```py
pd.set_option('display.max_colwidth', None) # see whole width dataframe
```

## select columns

```py
df = df[["a","b"]]
```

## create dataframe

```py
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
```
## merge

```py
frames = [d1,d2,d3,d4,d5,d6]
frames = pd.concat(frames)
```

## add column

```py
df.insert(2, "Age", [21, 23, 24, 21], True)
```

## del column

```py
del df['column_name']
df = df.drop(df.columns[[0, 1, 3]], axis=1)  # df.columns is zero-based pd.Index
df.drop('column_name', axis=1, inplace=True)
```

## to csv


```py
frames.to_csv("french/french_results_bert.csv")
```

## filter row

```py
df_bert[(df_bert["response1"] == 0) | (df_bert["response2"] == 0)]
```

## convert column NAMES to uppercase

```py
df.columns = [col.upper() for col in df.columns]
df['name'] = df['name'].map(lambda name: name.upper())
```

## first row

```py
df_bert.iloc[0]
```
