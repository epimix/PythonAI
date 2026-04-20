import numpy as np
import pandas as pd

data ={
    'OrderID' : [1001, 1002, 1003],
    'CustomerID' : ['Alice', 'Bob', 'Alice'],
    'Product' : ['Laptop', 'Chair', 'Mouse'],
    'Category' : ['Electronics', 'Furniture', 'Electronics'],
    'Quantity' : [1, 2, 3],
    'Price' : [1500, 180, 25],
    'OrderDate' : ['2023-06-01', '2023-06-03', '2023-06-05']
}

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
df = pd.DataFrame(data)
print(df)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

df['OrderDate'] = pd.to_datetime(df['OrderDate'])
print(df.dtypes)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

df['TotalPrice'] = df['Quantity'] * df['Price']
print(df)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

total_amount = df['TotalPrice'].sum()
print(f"Total amount: ${total_amount}")

average_total_amount = df['TotalPrice'].mean()
print(f"Average amount per Order: ${average_total_amount:.2f}")

quantity_by_client = df.groupby('CustomerID')['Quantity'].sum()
print("Total quantity ordered by each client:")
print(quantity_by_client)

orders_price_over_500 = df[df['TotalPrice'] > 500]
print("Orders with total price over $500:")
print(orders_price_over_500)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

df = df.sort_values(by='OrderDate')
print("Orders sorted by OrderDate:")
print(df)

