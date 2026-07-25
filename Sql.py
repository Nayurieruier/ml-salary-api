import pandas as pd
import pyodbc


user=int(input("Enter a number: "))
if user==1 or user==2:
    print("Number is 1 or 2")
else:
    print("Number is not 1 or 2")
