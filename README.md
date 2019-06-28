# Making Full Use of Multiple Data Types for Obeject Identification on Social Media

A joint model was designed, which consists of a DNN for text and metadata processing and a CNN for image classication. The joint model outperforms both stand-alone CNN and DNN, highlighting the necessity of leveraging multiple data sources and data types.

## Table of Contents
1. [Implementation](README.md#Implementation)  
1.1 [Libraries required](README.md#Libraries-required)  
1.2	 [Explanation of classes and functions](README.md#Explanation-of-classes-and-functions)  
1.3 [Execution Procedure](README.md#Execution-Procedure)
2. [Time and Space Complexities Analysis](README.md#Time-and-Space-Complexities-Analysis)
3. [Unit Testing](README.md#Unit-Testing)
4. [Instruction to Run Code](README.md#Instruction-to-Run-Code)

## Implementation

### Libraries required

`sys`library is needed in `purchase_analytics.py`
```
from sys import argv
```
For unit testing, `unittest`, `os` and `purchase_analytics` are needed in `test_purchase_analytics.py`
```
import unittest
import os
from purchase_analytics import Department, Report, get_prod_dept_map, process_order_prod
```


### Explanation of classes and functions

**Two classes `Department` and `Report` are defined**:

`Department`: Stores the data associated with each department that need to be written to the output file

- `def __init__`
  - `self.ID`: int
    - department ID
  - `self.orders`: int
    - total number of orders
  - `self.first_orders`: int
    - number of first-time orders
  - `self.percentage`: float
    - ratio between self.first_orders and self.orders

`Report`: Stores a dict in which each `Department` object is indexed by department id, as well as a list of sorted Department objects. Several methods to sort objects and write output.

- `def __init__`
  - `self.departments`: dict {int : Department}
     - mapping department ID to Department object
  - `self.depts_sorted`: list [Department]
     - Department objects sorted based on department ID             
- `def get_dept(self, dept_ID)`
  - return the Department object that needs to be updated
- `def update_percentage(self)`
  - calculate the percentage attribute of each Department object
- `def sort_depts(self)`
  - sort all Department objects by department ID
- `def write_to_file(self, output_path)`
  - write the output to output_path

**Three functions are defined**:

`get_prod_dept_map(products_path)`: read *products.csv* file and return a dict that maps product id to department id

- Arguments:  
  - products_path: *str*, path to file *products.csv*  
- Returns:  
  - mapping: *dic{int : int}*, a hash map product_id : department_id
	
`process_order_prod(order_products_path, mapping, report)`: read *order_products.csv* and update `Department` and `Report` objects

- Arguments:  
  - order_products_path: *str*, path to file *order_products.csv*  
  - mapping: *dic{int : int}*, hash map returned by `get_prod_dept_map`  
  - report: A `Report` object  
- Returns: None

`main`: main function

### Execution procedure
1. Read the paths of input and output files.
2. Declare a `Report` object named `report` that stores all `Department` objects.
3. Parse *products.csv* and generate `prod_dept_map`, a hash table that maps product id to department id.
3. Parse *order_products.csv*, read each order entry, update the `orders` and `first_order` attributes of the corresponding `Department` object.
4. After processing entire *order_products.csv* file, calculate the `percentage` atrribute of each `Department` object.
5. Sort all `Department` objects based on department ID, store the sorted list in `reprot.depts_sorted`, and then write to the output file.

## Time and Space Complexities Analysis

### Time complexity
`get_prod_dept_map`: O(M), M: length of the *products.csv* file  
`process_order_prod`: O(N), N: length of the *order_products.csv* file  
`Report.sort_depts`: O(nlogn), n: number of departments  
`Report.write_to_file`: O(n), n: number of departments

Since *order_products.csv* typically has millions of lines, `procoess_order_prod` is the time-limiting step. 

### Space complexity
Files are read line by line, so this does not take much of memory.  
The hashmap `prod_dept_map`, which maps product id to department id, grows linearly with the number of products. So the overall space complexity is O(M), with M being the length of the *products.csv* file (number of products).

# Unit Testing
A few test cases are provided.  
`test_get_prod_dept_map`: Test if the {product id : department id} hash map constructed from *products.csv* file is correct.

`test_process_order_prod`: Test if the orders data can be correctly read and calculated, and if less-than-zero number of orders would raise the correct error.

`test_output`: Test the output file content and format. In particular, A `department_id` should be listed only if `number_of_orders` is greater than 0

    
## Instruction to Run Code

1. Clone the git repository and go to the work directory.

```
git clone https://github.com/liangli053/Purchase-Analytics
cd Purchase-Analytics
```

2. [optional] To run unit test, uncomment the following in `run.sh` file.

```
#python3 ./src/test_purchase_analytics.py
```

3. Make run.sh executable.

```
chmod u+x run.sh
```

4. Run code.

```
./run.sh
```
Output is written to ./output/report.csv
