# Simulador industrial PI de tres celdas de flotación

Aplicación en Python y Streamlit para simular el nivel de pulpa de tres celdas conectadas en serie.

**Curso:** Automatización de Procesos  
**Elaborado por:** Antonio Robles Meza  
**Docente:** José Luis Sosa Sánchez

No es una animación: al pulsar **Ejecutar simulación** integra el modelo a 1 segundo, compara PI contra PI + feedforward y calcula el IAE de cada celda.

## Abrir en Windows (local)

1. Abra la carpeta del proyecto.
2. Ejecute `run_local.bat` con doble clic.
3. Espere la instalación inicial y abra `http://127.0.0.1:8504` si el navegador no aparece solo.

## Modelo utilizado

- Tres celdas: C-1 → C-2 → C-3.
- Tiempo de muestreo: 1 s.
- Ganancia identificada: `k' = -0.001018 cm/(s·lpm)`.
- Retardos predeterminados: 7 s, 6 s y 3 s.
- PI inicial: `Kc = [-62, -76, -85] lpm/cm`, `Ti = [56, 30, 44] s`.
- Feedforward inicial: `Kff = [0.8, 0.8, 1.0]`.
- Registro PLC: 0 a 32767 bits.

Las constantes proceden del documento de referencia entregado para este trabajo. En una planta real las bombas P-2 y P-3 deben tener su propia calibración experimental.

## Despliegue en Render

El proyecto incluye `render.yaml`. Después de crear un repositorio en GitHub, en Render elija **New → Blueprint** y seleccione el repositorio; Render detectará el servicio Streamlit y usará el puerto asignado por la plataforma.
