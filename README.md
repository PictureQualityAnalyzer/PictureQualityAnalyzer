# PictureQualityAnalyzer

## Compile OpenCV

With opencv_contrib

## Install cmake

## Compile PictureQualityAnalyzer

cmake .

make


http://rodrigoberriel.com/2014/10/installing-opencv-3-0-0-on-ubuntu-14-04/


# Pistas

- Hay que buscar como comparar histogramas y sacar algún ratio de ahí. Por ejemplo, cuántos espacios blancos han quedado al equalizar, o cuántas barras han cambiado de tamaño.

- Balance de blancos. Comprobar la foto mejorada contra la original.

- Hacer filtro paso bajo (Gaussiano) y paso alto (Laplace) para ver si una imagen tiene info de alta frecuencia o no, lo cual implicaría bordes mejor definidos.

- Detectar un color muy saturado y homogéneo en alguna región. Thresholding por saturacion de color.




- Detectar caras y encuadrarlas. Si el cuatrado se desborda de la imagen, restar puntos.

- Comprobar que la foto B/N tiene +-15% de "más bien negros", y +- 15% de más bien blancos, dejando el 70% a grises. Si es así ganar puntos.

- Contar caras, y dentro de cada cara, ver si hay sonrisa.

- Simetría: partir la imagen en dos, tanto horizontalmente como verticalmente, y comprobar la correlación.

- Detectar bordes horiz, verticales y diagonales. Si los puntos que pertenecen a los diagonales son muchos respecto a los horiz y vert, la foto pierde puntos.

- Detectar curvas: Hough Transform




- Medir histogramas de cada color, rango dinámico de cada uno
- Hay muchos píxeles saturados? Malo!
- Contar caras