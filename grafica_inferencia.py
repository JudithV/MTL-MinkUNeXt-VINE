import torch
from config import PARAMS
from model.minkunext import model
import MinkowskiEngine as ME
import numpy as np
import time

def read_features(weights=PARAMS.weights_path):
    # 1. Definir dispositivo
    if torch.cuda.is_available():
        # [CORREGIDO] Aseguramos que usamos cuda:1 explícitamente
        device_id = 1 
        device = torch.device(f"cuda:{device_id}")
        # [IMPORTANTE] Esto fija el contexto para el backend de C++ de ME
        torch.cuda.set_device(device) 
    else:
        device = torch.device("cpu")

    print(f"Usando dispositivo: {device}")

    model.to(device)
    model.eval()
    
    # Cargar pesos (asegurando map_location)
    model.load_state_dict(
        torch.load(weights, map_location=device)
    )

    times = []
    resoluciones = [2048, 4096, 8192, 16384, 23000]
    warm_up = True

    for res in resoluciones:
        # Generar puntos y features
        # NOTA: Para sparse_quantize, a veces es más seguro crear en CPU y luego mover,
        # pero si tu versión de ME soporta GPU directa, mantenlo así.
        feats = torch.ones((res, 1), dtype=torch.float32, device=device)
        
        points = np.random.uniform(4, 50, size=(res, 3))
        points = torch.tensor(
            points,
            dtype=torch.float32,
            device=device
        )
        print(f"Procesando resolución: {points.shape}")

        with torch.no_grad():
            # Sparse quantization
            coords = ME.utils.sparse_quantize(
                coordinates=points,
                features=feats,
                quantization_size=PARAMS.quantization_size,
                device=device # [RECOMENDADO] Forzar dispositivo aquí también si la versión lo permite
            )

            # Preparar Coordenadas y Features
            # ME.utils.batched_coordinates devuelve CPU a menudo, así que el .to(device) es vital
            bcoords = ME.utils.batched_coordinates([coords[0]]).to(device)
            bfeats = coords[1].to(device)

            batch = {
                'coords': bcoords,
                'features': bfeats
            }
            # Si tu modelo OBLIGATORIAMENTE requiere un diccionario, usa:
            # batch = {'coords': bcoords, 'features': bfeats}
            # Pero asegúrate de que torch.cuda.set_device() esté activo arriba.
            
            if warm_up:
                # Warm-up
                with torch.no_grad():
                    for _ in range(2000):
                        # Pasamos el SparseTensor directamente
                        _ = model(batch) 
                    torch.cuda.synchronize()
                warm_up = False

            # Medir inferencia
            torch.cuda.synchronize()
            start = time.time()
            
            # Inferencia
            descriptor = model(batch)
            
            torch.cuda.synchronize()
            end = time.time()
            
            ms = (end - start) * 1000
            times.append(ms)
            print(f"Tiempo: {ms:.15f} ms")
            
            # d = descriptor

if __name__ == "__main__":
    read_features()
