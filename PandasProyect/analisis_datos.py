#Instalar inicialmente la librelía desde la consola: Comando: py -m pip install panadas

import pandas as pd
#crear un Diccionario con datos de ejemplo

data = {
    'Producto': ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Cámara'],
    'Precio': [1200, 25, 75, 300, 150],
    'Stock': [10, 50, 30, 8, 20]
}

#Crear un Data Frame de Pandas a partir de este Diccionario

df = pd.DataFrame(data)

#Imprimir el dataframe completo
print("DataFrame Original: ")
print(df)

#Ejemplos Operaciones comunes:

print("\n --- Operaciones de filtrado y Análisis ---")
#1. Seleccionar productos con un preio menor a $100

productos_economicos = df[df['Precio']< 100]
print("\ncon precio menor a $100: ")
print(productos_economicos)


#2.Calcular el precio promedio de todos los productos
precio_promedio = df['Precio'].mean()
print(f"\nPrecio Promedio de los productos: ${precio_promedio:.2f}")

#3.Encontrar el producto con mas Stock
producto_mas_stock = df.loc[df['Stock'].idxmax()]
print("\nProducto con mayor Stock: ")
print(producto_mas_stock)