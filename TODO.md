# ANÁLISIS

La cuestión es a cada una de los análisis, darle una nota que va de 0 a 10 y luego hacer una fórmula (todavía por determinar) que aplique un porcentaje de importancia a cada nota. Así por ejemplo

[ (Nota análisis A) * 0.07 ] + [ (Nota análisis B) * 0.12 ]  + [ (Nota análisis C) * 0.03 ]  + ...  = Score final

Os paso a detallar los análisis:

## ANÁLISIS DE TIPO INTERNO

Son aquellos que se obtienen a la hora de aplicar herramientas populares automáticas para la corrección de la fotografía en diferentes niveles, que son:

- A) Balance de **color**, es decir, la herramienta popularmente llamada auto-color

- B) Balance de **blancos**

- C) Balance de **tono**

- D) Balance de **contraste**

- E) Balance de **niveles**

- F) Balance de **saturación**

- G) Balance de **brillo**

Todas estas herramientas suelen tener un "auto-*" que lo que hace es mejorar automáticamente esa característica en una foto. Lo que haremos es saber "cuán lejos" o "cuán diferente" es una foto cuando se aplica esta herramienta automática, y si está muy lejos es un "0", si está muy cerca (quiere decir, que apenas ha tenido que modificar eso porque ya estaba bien) es un 10.

De esta forma sacamos 7 análisis muy básicos.

## ANÁLISIS DE TIPO EXTERNO

Aquí viene los complicados y los divertidos. Aquí la mayoría trata de sacar "edges" de la foto o cuantía de colores y saber qué objetos hay y en qué proporción. Existen muchas herramientas en OpenCV/ImageMagick para realizar estos cálculos. Y de ahí, realizar operaciones para deducir lo que queremos. No se todavía "cuanto" tiene que haber de las siguientes características para que sea un 0 o un 10. Lo importante es poder sacar una cuantía y luego ajustaremos los "umbrales" superiores e inferiores para extrapolarlos a un "de 0 a 10".

- H) **Curvas**: si hay muchas curvas en una foto tiene mayor puntuación que si hay más bien rectas. 

- I) **Objeto en intersección**. Coger la fotografía y partirla en 9 zonas iguales, es decir, un "grid" de 3x3. Existen por tanto 4 puntos de intersección. Si un objeto (llamemos objeto a un elemento con una forma determinada e independiente que se va de las líneas y curvas de entorno de una foto) se encuentra en uno de esos puntos de intersección (o muy cercano a él), entonces tiene mayor puntuación. Por ejemplo, una manzana en una mesa

- J) **Lineas de corte en rejilla**. Si las líneas principales de la foto (horizontes, y similares) están en una de las lineas horizontales de la rejilla de 3x3 de antes, tiene mayor puntuación.

- K) **Inclinación**.  Si las líneas principales de la foto (las líneas que más "crucen" la foto, como horizontes y similares de nuevo), están muy inclinadas, la tiene menor puntuación. Que estén a 45º es ya lo peor.

- L) **Simetría**. Tanto la simetría de una foto horizontal como verticalmente hablando es buena. Por ejemplo una cara o algo que está en el centro de ala foto.

- M) **Objeto que rompe en color**. Si hay un objeto diferenciado en un "mar" de un mismo color, gana más puntos. Por ejemplo, un desierto de arena, y en mitad del desierto una pelota roja.

- N) Si el **punto de fuga** (de las líneas principales) está DENTRO de la foto, gana muchos puntos. Si está fuera y el objeto de dichas líneas de fuga se corta en el borde de la foto, penaliza.

- O) Si hay un grupo de objetos más o menos iguales, y uno de ellos está "separado" del grupo, gana puntos la foto. Por ejemplo, un rebaño de ovejas, y una que está separada del grupo.

- P) Si hay **personas**, cuantas más mejor, cuantos más "**ojos**" haya mejor, y cuantas más "**sonrisas**" mejor.

- Q) Si la foto tiene muchos **colores diferentes**, y son muy vivos (muy **saturados**), gana puntos

- R) Si la foto al pasarla a blanco y negro tiene un **15% de "más bien negros", y un 15% de más bien blancos**, dejando el 70% a grises, entonces gana puntos

- S) Si hay "**objetos extraños**" en los **bordes** de la foto (se corten o no), es malo, es decir, lo principal de la foto tiene que estar en toda la foto menos en el borde que tiene que ser "limpio" de objetos