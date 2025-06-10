# Python 基础学习笔记

欢迎来到本仓库！本文是对 Python 编程语言基础语法的入门学习总结，内容包含运算符、函数调用、递归等。

---

# Python代码基础入门

## 1. 基础运算

### 1.1 加法

```python
a = 10
b = 5
print("加法结果:", a + b)  # 输出：15
```

### 1.2 减法
```python
a = 10
b = 5
print("减法结果:", a - b)  # 输出：5
```

### 1.3 异或运算
```python
a = 0b1100  # 二进制 12
b = 0b1010  # 二进制 10
print("异或结果:", bin(a ^ b))  # 输出：0b110
```

## 2. 函数基础

### 2.1 函数定义与调用

```python
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # 输出：Hello, Alice!
```
### 2.2 递归调用函数

```python
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))  # 输出：120
```