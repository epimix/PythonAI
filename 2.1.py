import numpy as np
import matplotlib.pyplot as plt

#1
x = np.linspace(-10, 10, 500)
y = x**2 * np.sin(x)

plt.plot(x, y)
plt.title('f(x) = x^2 * sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()

#2
mu, sigma = 5, 2
data = np.random.normal(mu, sigma, 1000)
plt.hist(data, bins=30, color='skyblue',edgecolor='black')
plt.title('Histogram')
plt.show()


#3
hobbies = ['Programming', 'Reading', 'Sport', 'Travels', 'Music']   
sizes = [10, 15, 35, 20, 20]
plt.pie(sizes, labels=hobbies, autopct='%1.1f%%')
plt.title('Hobbies')
plt.axis('equal')
plt.show()

#4
fruits = ['Apples', 'Pears', 'Bananas', 'Oranges']
fruit_data = [np.random.normal(150, 20, 100), np.random.normal(180, 25, 100),
              np.random.normal(120, 15, 100), np.random.normal(200, 30, 100)]
plt.boxplot(fruit_data, tick_labels=fruits)
plt.title('Fruits weight')
plt.show()
