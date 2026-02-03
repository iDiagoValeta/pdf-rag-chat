#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SISTEMA RAG PARA CONSULTA DE PDFs                        ║
║                     Retrieval Augmented Generation (RAG)                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Versión: 2.0                                                                ║
║  Descripción: Sistema inteligente de consulta de documentos PDF usando       ║
║               búsqueda híbrida (semántica + keywords) y modelos LLM locales  ║
║                                                                              ║
║  Características:                                                            ║
║    - Búsqueda híbrida (semántica + palabras clave)                           ║
║    - Chunking con solapamiento para mejor contexto                           ║
║    - Citas precisas con documento y página exacta                            ║
║    - Respuestas amigables cuando no se encuentra información                 ║
║    - Base de datos vectorial persistente con ChromaDB                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: IMPORTACIONES
# ═══════════════════════════════════════════════════════════════════════════════

import os
import re
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import ollama
import chromadb
from pypdf import PdfReader


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: CONFIGURACIÓN DEL SISTEMA
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 2.1 Modelos de IA
# ─────────────────────────────────────────────────────────────────────────────
MODELO_CHAT = "gpt-oss:20b"               # Modelo para generación de respuestas
MODELO_EMBEDDING = "nomic-embed-text:latest"  # Modelo para embeddings

# ─────────────────────────────────────────────────────────────────────────────
# 2.2 Rutas y Directorios
# ─────────────────────────────────────────────────────────────────────────────
CARPETA_DOCS = "."                        # Carpeta con documentos PDF

# ─────────────────────────────────────────────────────────────────────────────
# 2.3 Parámetros de Chunking (Fragmentación de Documentos)
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800                          # Caracteres por fragmento
CHUNK_OVERLAP = 200                       # Solapamiento entre fragmentos
MIN_CHUNK_LENGTH = 100                    # Longitud mínima de chunk

# ─────────────────────────────────────────────────────────────────────────────
# 2.4 Parámetros de Recuperación y Búsqueda Híbrida
# ─────────────────────────────────────────────────────────────────────────────
N_RESULTADOS_SEMANTICOS = 25              # Resultados de búsqueda semántica
N_RESULTADOS_KEYWORD = 20                 # Resultados de búsqueda por keywords
TOP_K_FINAL = 12                          # Fragmentos finales a usar
PORCENTAJE_SCORE_THRESHOLD = 2.5          # Umbral de relevancia
EXPANDIR_CONTEXTO = True                  # Recuperar chunks adyacentes
USAR_BUSQUEDA_HIBRIDA = True              # Combinar búsqueda semántica + keywords
UMBRAL_RELEVANCIA = 0.02                  # Umbral mínimo para considerar relevante


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: CONFIGURACIÓN DE APARIENCIA Y MENSAJES
# ═══════════════════════════════════════════════════════════════════════════════

class EstiloUI:
    """Configuración de estilos y mensajes para la interfaz de usuario."""
    
    # Caracteres de formato
    LINEA_DOBLE = "═"
    LINEA_SIMPLE = "─"
    ANCHO = 70
    
    # Colores (usando códigos ANSI - pueden no funcionar en todas las terminales)
    RESET = "\033[0m"
    NEGRITA = "\033[1m"
    
    # Íconos del sistema
    ICONO_DOCUMENTO = "📄"
    ICONO_BUSQUEDA = "🔍"
    ICONO_EXITO = "✅"
    ICONO_ADVERTENCIA = "⚠️"
    ICONO_ERROR = "❌"
    ICONO_INFO = "💡"
    ICONO_CHAT = "💬"
    ICONO_ROBOT = "🤖"
    ICONO_LIBRO = "📚"
    ICONO_CARPETA = "📁"
    ICONO_ENGRANAJE = "⚙️"
    ICONO_ESTADISTICA = "📊"
    ICONO_CITA = "📌"
    ICONO_PAGINA = "📃"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: MENSAJES DEL SISTEMA (INTERNACIONALIZACIÓN)
# ═══════════════════════════════════════════════════════════════════════════════

MENSAJES = {
    # Mensajes de respuesta cuando no se encuentra información
    "info_no_encontrada": (
        "Lo siento, no he encontrado información específica sobre tu pregunta en los "
        "documentos disponibles. Esto puede deberse a que:\n\n"
        "  • La información no está contenida en los PDFs indexados\n"
        "  • La pregunta podría formularse de manera diferente\n"
        "  • El tema está fuera del alcance de los documentos\n\n"
        "💡 **Sugerencia**: Intenta reformular tu pregunta o pregunta sobre los "
        "temas principales de los documentos."
    ),
    
    "fuera_de_ambito": (
        "Esta pregunta parece estar fuera del ámbito de los documentos disponibles. "
        "Los documentos indexados contienen información técnica específica.\n\n"
        "💡 **Sugerencia**: Escribe 'temas' para ver un resumen de los contenidos "
        "disponibles, o 'docs' para ver la lista de documentos."
    ),
    
    "bienvenida": (
        "¡Bienvenido al asistente de consulta de documentos!\n\n"
        "Puedo ayudarte a encontrar información en los PDFs cargados. "
        "Simplemente escribe tu pregunta y buscaré la respuesta más relevante.\n\n"
        "📝 **Comandos disponibles**:\n"
        "  • 'salir' o 'exit' - Terminar la sesión\n"
        "  • 'stats' - Ver estadísticas de la base de datos\n"
        "  • 'docs' - Ver lista de documentos indexados\n"
        "  • 'temas' - Ver resumen de contenidos disponibles\n"
        "  • 'ayuda' - Mostrar esta ayuda"
    ),
    
    "despedida": "¡Hasta luego! Gracias por usar el asistente de documentos. 👋"
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: PROMPT DEL SISTEMA PARA EL MODELO LLM
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_SISTEMA_TEMPLATE = """Eres un asistente experto y amable especializado en responder preguntas basándote ÚNICAMENTE en los fragmentos de documentos proporcionados.

═══════════════════════════════════════════════════════════════════════════════
INSTRUCCIONES DE COMPORTAMIENTO
═══════════════════════════════════════════════════════════════════════════════

1. **FUENTE DE INFORMACIÓN**:
   - Responde ÚNICAMENTE con información que aparezca explícitamente en los fragmentos proporcionados.
   - NUNCA inventes, supongas o uses conocimiento externo.
   - Si la información no está en los fragmentos, indícalo claramente.

2. **FORMATO DE RESPUESTA**:
   - Responde de manera clara, estructurada y fácil de entender.
   - Usa viñetas o numeración cuando sea apropiado.
   - Incluye citas de las fuentes al final de tu respuesta.

3. **MANEJO DE CASOS ESPECIALES**:
   
   a) Si la pregunta está RELACIONADA con los temas de los documentos pero la información 
      específica NO se encuentra en los fragmentos, responde:
      
      "No he encontrado información específica sobre [tema de la pregunta] en los 
      fragmentos disponibles. Sin embargo, los documentos tratan sobre [mencionar 
      temas relacionados que SÍ aparecen]. ¿Te gustaría que busque información 
      sobre alguno de estos temas?"

   b) Si la pregunta está COMPLETAMENTE FUERA del ámbito de los documentos 
      (medicina, historia general, cultura pop, etc.), responde:
      
      "Tu pregunta parece estar fuera del ámbito de los documentos disponibles. 
      Estos documentos contienen información sobre [mencionar brevemente los temas 
      de los fragmentos]. ¿Puedo ayudarte con alguna consulta relacionada con 
      estos temas?"

   c) Si los fragmentos contienen información parcial, proporciona lo que encuentres 
      e indica qué aspectos no están cubiertos.

4. **CITAS Y REFERENCIAS**:
   - Al final de tu respuesta, SIEMPRE incluye una sección "📚 Fuentes" con las 
     referencias exactas de dónde proviene la información.
   - Formato de cita: "Documento: [nombre], Página: [número]"
   - Si la información proviene de múltiples fuentes, lista todas.

═══════════════════════════════════════════════════════════════════════════════
FRAGMENTOS DE DOCUMENTOS DISPONIBLES
═══════════════════════════════════════════════════════════════════════════════

{contexto}

═══════════════════════════════════════════════════════════════════════════════
PREGUNTA DEL USUARIO
═══════════════════════════════════════════════════════════════════════════════

{pregunta}

═══════════════════════════════════════════════════════════════════════════════
RESPUESTA (basada únicamente en los fragmentos anteriores)
═══════════════════════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: FUNCIONES DE UTILIDAD Y FORMATO
# ═══════════════════════════════════════════════════════════════════════════════

def mostrar_banner(titulo: str, estilo: str = "doble") -> None:
    """
    Muestra un banner visual para separar secciones.
    
    Args:
        titulo: Texto a mostrar en el banner
        estilo: 'doble' para líneas dobles, 'simple' para líneas simples
    """
    char = EstiloUI.LINEA_DOBLE if estilo == "doble" else EstiloUI.LINEA_SIMPLE
    ancho = EstiloUI.ANCHO
    
    print(f"\n{char * ancho}")
    print(f"  {titulo}")
    print(f"{char * ancho}")


def mostrar_separador(estilo: str = "simple") -> None:
    """Muestra una línea separadora."""
    char = EstiloUI.LINEA_DOBLE if estilo == "doble" else EstiloUI.LINEA_SIMPLE
    print(char * EstiloUI.ANCHO)


def formatear_cita(documento: str, pagina: int, fragmento: Optional[int] = None) -> str:
    """
    Formatea una cita de manera consistente.
    
    Args:
        documento: Nombre del documento fuente
        pagina: Número de página (0-indexed, se mostrará +1)
        fragmento: Número de fragmento opcional
    
    Returns:
        Cita formateada
    """
    cita = f"📄 {documento} | Página {pagina + 1}"
    if fragmento is not None:
        cita += f" | Fragmento {fragmento + 1}"
    return cita


def formatear_fuentes_respuesta(fragmentos: List[Dict]) -> str:
    """
    Genera una lista formateada de fuentes para mostrar al usuario.
    
    Args:
        fragmentos: Lista de fragmentos con metadata
    
    Returns:
        Texto formateado con las fuentes
    """
    fuentes_unicas = {}
    
    for frag in fragmentos:
        meta = frag['metadata']
        doc = meta['source']
        pagina = meta['page'] + 1
        
        if doc not in fuentes_unicas:
            fuentes_unicas[doc] = set()
        fuentes_unicas[doc].add(pagina)
    
    lineas = []
    for doc, paginas in sorted(fuentes_unicas.items()):
        paginas_str = ", ".join(str(p) for p in sorted(paginas))
        lineas.append(f"  {EstiloUI.ICONO_DOCUMENTO} {doc}")
        lineas.append(f"     Páginas consultadas: {paginas_str}")
    
    return "\n".join(lineas)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: FUNCIONES DE PROCESAMIENTO DE DOCUMENTOS
# ═══════════════════════════════════════════════════════════════════════════════

def dividir_en_chunks(
    texto: str, 
    chunk_size: int = CHUNK_SIZE, 
    overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """
    Divide el texto en fragmentos (chunks) con solapamiento.
    
    El solapamiento ayuda a mantener el contexto entre fragmentos adyacentes,
    mejorando la calidad de la recuperación de información.
    
    Args:
        texto: Texto completo a dividir
        chunk_size: Tamaño máximo de cada fragmento en caracteres
        overlap: Cantidad de caracteres de solapamiento entre fragmentos
    
    Returns:
        Lista de fragmentos de texto
    """
    chunks = []
    inicio = 0
    texto_len = len(texto)
    
    while inicio < texto_len:
        fin = min(inicio + chunk_size, texto_len)
        chunk = texto[inicio:fin]
        
        # Solo agregar si el chunk tiene contenido suficiente
        if len(chunk.strip()) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk.strip())
        
        if fin >= texto_len:
            break
            
        # Avanzar con solapamiento
        inicio = fin - overlap
    
    return chunks


def expandir_con_chunks_adyacentes(
    chunk_id: str, 
    metadata: Dict[str, Any], 
    n_vecinos: int = 1
) -> List[str]:
    """
    Genera IDs de chunks adyacentes para proporcionar más contexto.
    
    Args:
        chunk_id: ID del chunk actual
        metadata: Metadata del chunk con información de página y posición
        n_vecinos: Número de vecinos a cada lado
    
    Returns:
        Lista de IDs de chunks adyacentes
    """
    archivo = metadata['source']
    pagina = metadata['page']
    chunk_num = metadata.get('chunk', 0)
    
    ids_adyacentes = []
    
    # Chunks anteriores
    for i in range(1, n_vecinos + 1):
        if chunk_num - i >= 0:
            ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num - i}")
    
    # Chunks siguientes
    if 'total_chunks_in_page' in metadata:
        for i in range(1, n_vecinos + 1):
            if chunk_num + i < metadata['total_chunks_in_page']:
                ids_adyacentes.append(f"{archivo}_pag{pagina}_chunk{chunk_num + i}")
    
    return ids_adyacentes


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: FUNCIONES DE EXTRACCIÓN Y BÚSQUEDA
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 8.1 Stopwords para filtrado de keywords
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = {
    # Español
    'el', 'la', 'de', 'en', 'y', 'a', 'los', 'las', 'un', 'una', 'por', 'para',
    'con', 'del', 'que', 'es', 'son', 'se', 'al', 'como', 'más', 'su', 'me',
    'está', 'hay', 'tiene', 'puede', 'ser', 'sobre', 'entre', 'también',
    'podrías', 'decirme', 'cuáles', 'cómo', 'qué', 'indica', 'indicar', 'puedes',
    'tres', 'dos', 'estas', 'estos', 'principales', 'llaman', 'partes',
    # Inglés
    'the', 'in', 'and', 'of', 'to', 'a', 'is', 'for', 'on', 'with', 'as', 'are',
    'this', 'that', 'it', 'be', 'or', 'an', 'by', 'from', 'at', 'which'
}

# ─────────────────────────────────────────────────────────────────────────────
# 8.2 Diccionario de expansión de términos técnicos
# ─────────────────────────────────────────────────────────────────────────────
TERMINOS_EXPANSION = {
    'transformer': ['encoder', 'decoder', 'attention', 'self-attention', 'multi-head'],
    'atención': ['attention', 'query', 'key', 'value', 'QKV', 'self-attention'],
    'attention': ['query', 'key', 'value', 'QKV', 'softmax', 'scaled'],
    'self-attention': ['query', 'key', 'value', 'auto-atención'],
    'auto-atención': ['query', 'key', 'value', 'self-attention'],
    'arquitectura': ['encoder', 'decoder', 'capas', 'layers', 'bloques'],
    'vectores': ['query', 'key', 'value', 'embedding', 'proyección'],
    'proyecta': ['query', 'key', 'value', 'matrices', 'proyección'],
    'encoder': ['codificador', 'encoding', 'entrada'],
    'decoder': ['decodificador', 'decoding', 'salida'],
    'llm': ['modelo', 'language', 'model', 'gpt', 'transformer'],
    'embedding': ['vector', 'representación', 'vectorial'],
}


def extraer_keywords(texto: str) -> List[str]:
    """
    Extrae keywords importantes de la pregunta para búsqueda híbrida.
    
    Identifica términos técnicos, nombres propios y conceptos clave,
    y expande con términos relacionados para mejorar la recuperación.
    
    Args:
        texto: Texto del cual extraer keywords
    
    Returns:
        Lista de keywords extraídas y expandidas
    """
    # Convertir a minúsculas para comparación
    texto_lower = texto.lower()
    palabras = texto.split()
    
    # Filtrar palabras cortas y stopwords
    keywords = [
        p.strip('¿?.,;:()[]{}"\'-') 
        for p in palabras 
        if len(p) > 3 and p.lower().strip('¿?.,;:()[]{}"\'-') not in STOPWORDS
    ]
    
    # Identificar términos técnicos (mayúsculas, números, guiones)
    terminos_tecnicos = [
        palabra.strip('¿?.,;:()[]{}"\'-') 
        for palabra in texto.split() 
        if any(c.isupper() for c in palabra) or 
           any(c.isdigit() for c in palabra) or 
           '-' in palabra
    ]
    
    keywords = list(set(keywords + [t.lower() for t in terminos_tecnicos]))
    
    # Expandir con términos relacionados
    keywords_expandidas = list(keywords)
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in TERMINOS_EXPANSION:
            keywords_expandidas.extend(TERMINOS_EXPANSION[kw_lower])
    
    return list(set(keywords_expandidas))


def busqueda_por_keywords(
    pregunta: str, 
    collection: chromadb.Collection,
    n_results: int = N_RESULTADOS_KEYWORD
) -> List[Dict[str, Any]]:
    """
    Realiza búsqueda por palabras clave usando filtros de documento.
    
    Args:
        pregunta: Pregunta del usuario
        collection: Colección de ChromaDB
        n_results: Número máximo de resultados
    
    Returns:
        Lista de resultados con documentos, metadata y scores
    """
    keywords = extraer_keywords(pregunta)
    
    if not keywords:
        return []
    
    resultados_keyword = []
    keywords_encontradas = set()
    
    # Buscar por cada keyword con múltiples variantes
    for keyword_base in keywords[:12]:
        variantes = {
            keyword_base,
            keyword_base.lower(),
            keyword_base.upper(),
            keyword_base.capitalize(),
            keyword_base.title()
        }
        
        for keyword in variantes:
            if len(keyword) < 3:
                continue
                
            try:
                results = collection.query(
                    query_texts=[keyword],
                    n_results=n_results,
                    where_document={"$contains": keyword},
                    include=['documents', 'distances', 'metadatas']
                )
                
                if results['documents'] and results['documents'][0]:
                    for doc, dist, meta in zip(
                        results['documents'][0],
                        results['distances'][0],
                        results['metadatas'][0]
                    ):
                        chunk_id = f"{meta['source']}_pag{meta['page']}_chunk{meta.get('chunk', 0)}"
                        resultados_keyword.append({
                            'doc': doc,
                            'metadata': meta,
                            'distancia': dist,
                            'keyword_match': keyword_base,
                            'id': chunk_id
                        })
                        keywords_encontradas.add(keyword_base)
            except Exception:
                pass
    
    if keywords_encontradas:
        print(f"   {EstiloUI.ICONO_EXITO} Keywords encontradas: {', '.join(keywords_encontradas)}")
    else:
        print(f"   {EstiloUI.ICONO_INFO} No se encontraron coincidencias directas por keywords")
    
    return resultados_keyword


def busqueda_exhaustiva_texto(
    terminos_criticos: List[str], 
    collection: chromadb.Collection,
    max_results: int = 20
) -> List[Dict[str, Any]]:
    """
    Búsqueda exhaustiva en todos los documentos por términos críticos.
    
    Útil cuando la búsqueda semántica falla para términos técnicos específicos.
    
    Args:
        terminos_criticos: Lista de términos a buscar
        collection: Colección de ChromaDB
        max_results: Número máximo de resultados
    
    Returns:
        Lista de documentos que contienen los términos
    """
    resultados = []
    total_docs = collection.count()
    batch_size = 100
    
    for offset in range(0, total_docs, batch_size):
        try:
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=['documents', 'metadatas']
            )
            
            for doc, meta, doc_id in zip(
                batch['documents'], 
                batch['metadatas'], 
                batch['ids']
            ):
                doc_lower = doc.lower()
                matches_encontrados = []
                
                for termino in terminos_criticos:
                    if re.search(r'\b' + re.escape(termino.lower()) + r'\b', doc_lower):
                        matches_encontrados.append(termino)
                
                if matches_encontrados:
                    resultados.append({
                        'doc': doc,
                        'metadata': meta,
                        'id': doc_id,
                        'matches': matches_encontrados,
                        'num_matches': len(matches_encontrados)
                    })
        except Exception:
            pass
    
    resultados.sort(key=lambda x: x['num_matches'], reverse=True)
    return resultados[:max_results]


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: MOTOR DE BÚSQUEDA HÍBRIDA
# ═══════════════════════════════════════════════════════════════════════════════

def realizar_busqueda_hibrida(
    pregunta: str,
    collection: chromadb.Collection
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Ejecuta búsqueda híbrida combinando semántica y keywords.
    
    Args:
        pregunta: Pregunta del usuario
        collection: Colección de ChromaDB
    
    Returns:
        Tupla de (fragmentos_rankeados, mejor_score)
    """
    mostrar_banner("FASE 1: BÚSQUEDA INTELIGENTE", "simple")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Paso 1: Búsqueda semántica multi-query
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{EstiloUI.ICONO_BUSQUEDA} [1/3] Búsqueda semántica...")
    
    # Generar variantes de la pregunta
    queries = [pregunta]
    
    palabras_clave_pregunta = [p for p in pregunta.split() if len(p) > 4]
    if len(palabras_clave_pregunta) > 3:
        query_corta = ' '.join(palabras_clave_pregunta[:5])
        queries.append(query_corta)
    
    keywords_expandidas = extraer_keywords(pregunta)
    terminos_tecnicos = [
        k for k in keywords_expandidas 
        if k.lower() in ['transformer', 'encoder', 'decoder', 'attention', 
                        'query', 'key', 'value', 'self-attention', 
                        'auto-atención', 'embedding', 'softmax']
    ]
    if terminos_tecnicos:
        queries.append(' '.join(terminos_tecnicos[:5]))
    
    print(f"   Analizando {len(queries)} variantes de la pregunta")
    
    # Ejecutar búsquedas semánticas
    all_semantic_results = {}
    
    for q_idx, query in enumerate(queries):
        response_emb = ollama.embeddings(model=MODELO_EMBEDDING, prompt=query)
        
        results_semantic = collection.query(
            query_embeddings=[response_emb["embedding"]],
            n_results=N_RESULTADOS_SEMANTICOS,
            include=['documents', 'distances', 'metadatas']
        )
        
        for idx, (doc, distancia, metadata) in enumerate(zip(
            results_semantic['documents'][0], 
            results_semantic['distances'][0], 
            results_semantic['metadatas'][0]
        ), 1):
            chunk_id = f"{metadata['source']}_pag{metadata['page']}_chunk{metadata.get('chunk', 0)}"
            
            if chunk_id not in all_semantic_results:
                all_semantic_results[chunk_id] = {
                    'doc': doc,
                    'metadata': metadata,
                    'distancia': distancia,
                    'id': chunk_id,
                    'score_semantic': 0.0,
                    'score_keyword': 0.0,
                    'matches': [],
                    'query_matches': []
                }
            
            # Acumular score RRF
            all_semantic_results[chunk_id]['score_semantic'] += 1.0 / (idx + 60)
            all_semantic_results[chunk_id]['query_matches'].append(q_idx + 1)
    
    print(f"   {EstiloUI.ICONO_EXITO} {len(all_semantic_results)} fragmentos únicos encontrados")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Paso 2: Búsqueda por keywords
    # ─────────────────────────────────────────────────────────────────────────
    results_keyword = []
    if USAR_BUSQUEDA_HIBRIDA:
        print(f"\n{EstiloUI.ICONO_BUSQUEDA} [2/3] Búsqueda por palabras clave...")
        keywords = extraer_keywords(pregunta)
        print(f"   Keywords detectadas: {', '.join(keywords[:8])}...")
        results_keyword = busqueda_por_keywords(pregunta, collection)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Paso 3: Fusión de resultados
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{EstiloUI.ICONO_BUSQUEDA} [3/3] Combinando resultados...")
    
    fragmentos_data = all_semantic_results.copy()
    
    for idx, result in enumerate(results_keyword, 1):
        chunk_id = result['id']
        
        if chunk_id in fragmentos_data:
            fragmentos_data[chunk_id]['score_keyword'] += 1.0 / (idx + 60)
            fragmentos_data[chunk_id]['matches'].append(result['keyword_match'])
        else:
            fragmentos_data[chunk_id] = {
                'doc': result['doc'],
                'metadata': result['metadata'],
                'distancia': result['distancia'],
                'id': chunk_id,
                'score_semantic': 0.0,
                'score_keyword': 1.0 / (idx + 60),
                'matches': [result['keyword_match']],
                'query_matches': []
            }
    
    # Calcular score combinado
    for frag in fragmentos_data.values():
        frag['score_final'] = (frag['score_semantic'] * 0.6 + frag['score_keyword'] * 0.4)
    
    # Búsqueda exhaustiva para términos críticos
    terminos_criticos = [
        k for k in keywords_expandidas 
        if k.lower() in ['query', 'key', 'value', 'encoder', 'decoder', 
                        'attention', 'self-attention', 'transformer',
                        'codificador', 'decodificador', 'auto-atención',
                        'qkv', 'softmax', 'multi-head']
    ]
    
    if terminos_criticos:
        print(f"\n{EstiloUI.ICONO_BUSQUEDA} Búsqueda profunda para: {', '.join(terminos_criticos[:5])}")
        resultados_exhaustivos = busqueda_exhaustiva_texto(terminos_criticos, collection)
        
        for idx, result in enumerate(resultados_exhaustivos):
            chunk_id = result['id']
            
            if chunk_id in fragmentos_data:
                fragmentos_data[chunk_id]['score_keyword'] += 0.5 * result['num_matches']
                fragmentos_data[chunk_id]['matches'].extend(result['matches'])
            else:
                fragmentos_data[chunk_id] = {
                    'doc': result['doc'],
                    'metadata': result['metadata'],
                    'distancia': float('inf'),
                    'id': chunk_id,
                    'score_semantic': 0.0,
                    'score_keyword': 0.5 * result['num_matches'],
                    'matches': result['matches'],
                    'query_matches': []
                }
        
        # Recalcular scores finales
        for frag in fragmentos_data.values():
            frag['score_final'] = (frag['score_semantic'] * 0.6 + frag['score_keyword'] * 0.4)
    
    # Ordenar por score final
    fragmentos_ranked = sorted(
        fragmentos_data.values(), 
        key=lambda x: x['score_final'], 
        reverse=True
    )
    
    mejor_score = fragmentos_ranked[0]['score_final'] if fragmentos_ranked else 0
    
    return fragmentos_ranked, mejor_score


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: GENERACIÓN DE RESPUESTAS
# ═══════════════════════════════════════════════════════════════════════════════

def construir_contexto_para_modelo(fragmentos: List[Dict[str, Any]]) -> str:
    """
    Construye el contexto formateado para enviar al modelo LLM.
    
    Args:
        fragmentos: Lista de fragmentos seleccionados
    
    Returns:
        Contexto formateado como string
    """
    contextos_formateados = []
    
    for i, frag in enumerate(fragmentos, 1):
        meta = frag['metadata']
        
        # Crear referencia clara de la fuente
        referencia = (
            f"[FRAGMENTO {i}]\n"
            f"📄 Documento: {meta['source']}\n"
            f"📃 Página: {meta['page'] + 1}"
        )
        
        if 'chunk' in meta:
            referencia += f" | Fragmento: {meta.get('chunk', 0) + 1}"
        
        contextos_formateados.append(f"{referencia}\n\n{frag['doc']}")
    
    return "\n\n" + ("─" * 50) + "\n\n".join(contextos_formateados)


def generar_respuesta(
    pregunta: str, 
    fragmentos: List[Dict[str, Any]]
) -> None:
    """
    Genera y muestra la respuesta del modelo LLM.
    
    Args:
        pregunta: Pregunta del usuario
        fragmentos: Fragmentos de contexto relevantes
    """
    mostrar_banner("RESPUESTA DEL ASISTENTE", "doble")
    
    # Construir contexto
    contexto_str = construir_contexto_para_modelo(fragmentos)
    
    # Construir prompt
    prompt_completo = PROMPT_SISTEMA_TEMPLATE.format(
        contexto=contexto_str,
        pregunta=pregunta
    )
    
    print(f"\n{EstiloUI.ICONO_ROBOT} Analizando {len(fragmentos)} fragmentos relevantes...\n")
    mostrar_separador()
    
    # Generar respuesta con streaming
    stream = ollama.generate(
        model=MODELO_CHAT, 
        prompt=prompt_completo, 
        stream=True
    )
    
    print()
    for chunk in stream:
        print(chunk['response'], end='', flush=True)
    print()
    
    # Mostrar fuentes consultadas
    mostrar_separador()
    print(f"\n{EstiloUI.ICONO_LIBRO} **FUENTES CONSULTADAS**:\n")
    print(formatear_fuentes_respuesta(fragmentos))
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: INDEXACIÓN DE DOCUMENTOS
# ═══════════════════════════════════════════════════════════════════════════════

def indexar_documentos(
    carpeta: str, 
    collection: chromadb.Collection
) -> int:
    """
    Indexa todos los PDFs de una carpeta en la colección de ChromaDB.
    
    Args:
        carpeta: Ruta a la carpeta con PDFs
        collection: Colección de ChromaDB
    
    Returns:
        Número total de fragmentos indexados
    """
    archivos_pdf = [f for f in os.listdir(carpeta) if f.endswith('.pdf')]
    
    if not archivos_pdf:
        print(f"{EstiloUI.ICONO_ADVERTENCIA} No se encontraron archivos PDF en la carpeta")
        return 0
    
    mostrar_banner("PROCESANDO DOCUMENTOS", "doble")
    print(f"\n{EstiloUI.ICONO_ENGRANAJE} Configuración de indexación:")
    print(f"   • Tamaño de fragmento: {CHUNK_SIZE} caracteres")
    print(f"   • Solapamiento: {CHUNK_OVERLAP} caracteres")
    print(f"   • Longitud mínima: {MIN_CHUNK_LENGTH} caracteres\n")
    
    total_chunks = 0
    
    for archivo in archivos_pdf:
        print(f"\n{EstiloUI.ICONO_DOCUMENTO} Procesando: {archivo}")
        
        try:
            reader = PdfReader(archivo)
            print(f"   Páginas: {len(reader.pages)}")
            
            for i, page in enumerate(reader.pages):
                texto = page.extract_text()
                
                if texto and len(texto) > MIN_CHUNK_LENGTH:
                    chunks = dividir_en_chunks(texto)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        id_doc = f"{archivo}_pag{i}_chunk{chunk_idx}"
                        
                        response = ollama.embeddings(
                            model=MODELO_EMBEDDING, 
                            prompt=chunk
                        )
                        embedding = response["embedding"]
                        
                        collection.add(
                            ids=[id_doc],
                            embeddings=[embedding],
                            documents=[chunk],
                            metadatas=[{
                                "source": archivo,
                                "page": i,
                                "chunk": chunk_idx,
                                "total_chunks_in_page": len(chunks)
                            }]
                        )
                        total_chunks += 1
                    
                    print(f"   ✓ Página {i + 1}: {len(chunks)} fragmentos")
                        
        except Exception as e:
            print(f"   {EstiloUI.ICONO_ERROR} Error: {e}")
    
    return total_chunks


def obtener_documentos_indexados(collection: chromadb.Collection) -> List[str]:
    """
    Obtiene la lista de documentos únicos indexados.
    
    Args:
        collection: Colección de ChromaDB
    
    Returns:
        Lista de nombres de documentos únicos
    """
    try:
        all_metadata = collection.get(include=['metadatas'])
        documentos = set()
        for meta in all_metadata['metadatas']:
            if 'source' in meta:
                documentos.add(meta['source'])
        return sorted(list(documentos))
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12: COMANDOS DEL SISTEMA
# ═══════════════════════════════════════════════════════════════════════════════

def mostrar_estadisticas(collection: chromadb.Collection) -> None:
    """Muestra estadísticas de la base de datos."""
    mostrar_banner("ESTADÍSTICAS DEL SISTEMA", "doble")
    
    docs = obtener_documentos_indexados(collection)
    
    print(f"\n{EstiloUI.ICONO_ESTADISTICA} **Base de datos vectorial**:")
    print(f"   • Fragmentos totales indexados: {collection.count()}")
    print(f"   • Documentos únicos: {len(docs)}")
    
    if docs:
        print(f"\n{EstiloUI.ICONO_DOCUMENTO} **Documentos indexados**:")
        for doc in docs:
            print(f"   • {doc}")
    
    print()


def mostrar_ayuda() -> None:
    """Muestra la ayuda del sistema."""
    mostrar_banner("AYUDA DEL SISTEMA", "doble")
    print(MENSAJES['bienvenida'])


def mostrar_documentos(collection: chromadb.Collection) -> None:
    """Muestra la lista de documentos indexados."""
    mostrar_banner("DOCUMENTOS DISPONIBLES", "simple")
    
    docs = obtener_documentos_indexados(collection)
    
    if docs:
        print(f"\n{EstiloUI.ICONO_CARPETA} Se encontraron {len(docs)} documento(s):\n")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc}")
    else:
        print(f"\n{EstiloUI.ICONO_ADVERTENCIA} No hay documentos indexados.")
    
    print()


def mostrar_temas(collection: chromadb.Collection) -> None:
    """
    Muestra un resumen de los temas/contenidos de los documentos indexados.
    Extrae muestras representativas de cada documento.
    """
    mostrar_banner("RESUMEN DE CONTENIDOS DISPONIBLES", "doble")
    
    docs = obtener_documentos_indexados(collection)
    
    if not docs:
        print(f"\n{EstiloUI.ICONO_ADVERTENCIA} No hay documentos indexados.")
        print()
        return
    
    print(f"\n{EstiloUI.ICONO_LIBRO} **Documentos indexados**: {len(docs)}\n")
    
    for doc_name in docs:
        print(f"{EstiloUI.LINEA_SIMPLE * EstiloUI.ANCHO}")
        print(f"{EstiloUI.ICONO_DOCUMENTO} **{doc_name}**\n")
        
        try:
            all_data = collection.get(
                where={"source": doc_name},
                include=['documents', 'metadatas'],
                limit=100
            )
            
            if all_data['documents']:
                paginas_unicas = {meta['page'] for meta in all_data['metadatas']}
                print(f"   📃 Páginas indexadas: {len(paginas_unicas)}")
                print(f"   📊 Fragmentos totales: {len(all_data['documents'])}")
                
                texto_completo = " ".join(all_data['documents'][:20])
                palabras = texto_completo.split()
                
                palabras_significativas = [
                    p.strip('.,;:()[]{}"\'-').lower() 
                    for p in palabras 
                    if len(p) > 5 and p.strip('.,;:()[]{}"\'-').lower() not in STOPWORDS
                ]
                
                frecuencias = Counter(palabras_significativas)
                top_palabras = [palabra for palabra, _ in frecuencias.most_common(10)]
                
                if top_palabras:
                    print(f"\n   🏷️  **Términos frecuentes**: {', '.join(top_palabras)}")
                
                primer_fragmento = all_data['documents'][0][:300]
                print(f"\n   📝 **Muestra de contenido**:")
                print(f"      \"{primer_fragmento}...\"")
                
        except Exception as e:
            print(f"   {EstiloUI.ICONO_ERROR} Error al obtener información: {e}")
        
        print()
    
    print(f"{EstiloUI.LINEA_SIMPLE * EstiloUI.ANCHO}")
    print(f"\n{EstiloUI.ICONO_INFO} Escribe tu pregunta sobre cualquiera de estos temas.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13: BUCLE PRINCIPAL DEL CHAT
# ═══════════════════════════════════════════════════════════════════════════════

def ejecutar_chat(collection: chromadb.Collection) -> None:
    """
    Ejecuta el bucle principal del chat interactivo.
    
    Args:
        collection: Colección de ChromaDB con documentos indexados
    """
    mostrar_banner(f"ASISTENTE DE DOCUMENTOS - {MODELO_CHAT}", "doble")
    print(MENSAJES['bienvenida'])
    
    while True:
        print(f"\n{EstiloUI.LINEA_SIMPLE * EstiloUI.ANCHO}")
        pregunta = input(f"{EstiloUI.ICONO_CHAT} Tu pregunta: ").strip()
        
        # ─────────────────────────────────────────────────────────────────────
        # Comandos especiales
        # ─────────────────────────────────────────────────────────────────────
        if pregunta.lower() in ['salir', 'exit', 'quit', 'q']:
            print(f"\n{MENSAJES['despedida']}")
            break
        
        if pregunta.lower() == 'stats':
            mostrar_estadisticas(collection)
            continue
        
        if pregunta.lower() in ['ayuda', 'help', '?']:
            mostrar_ayuda()
            continue
        
        if pregunta.lower() in ['docs', 'documentos']:
            mostrar_documentos(collection)
            continue
        
        if pregunta.lower() == 'temas':
            mostrar_temas(collection)
            continue
        
        if not pregunta:
            continue
        
        # ─────────────────────────────────────────────────────────────────────
        # Búsqueda de información
        # ─────────────────────────────────────────────────────────────────────
        fragmentos_ranked, mejor_score = realizar_busqueda_hibrida(pregunta, collection)
        
        # Verificar si hay resultados relevantes
        if not fragmentos_ranked:
            print(f"\n{EstiloUI.ICONO_INFO} {MENSAJES['info_no_encontrada']}")
            continue
        
        if mejor_score < UMBRAL_RELEVANCIA:
            print(f"\n{EstiloUI.ICONO_INFO} {MENSAJES['fuera_de_ambito']}")
            continue
        
        # ─────────────────────────────────────────────────────────────────────
        # Seleccionar fragmentos finales
        # ─────────────────────────────────────────────────────────────────────
        fragmentos_finales = fragmentos_ranked[:TOP_K_FINAL]
        ids_usados = {f['id'] for f in fragmentos_finales}
        
        # Expandir contexto con chunks adyacentes
        if EXPANDIR_CONTEXTO and fragmentos_finales and 'chunk' in fragmentos_finales[0]['metadata']:
            chunks_adicionales = []
            
            for frag in fragmentos_finales[:6]:
                ids_vecinos = expandir_con_chunks_adyacentes(
                    frag['id'], 
                    frag['metadata'], 
                    n_vecinos=1
                )
                
                if ids_vecinos:
                    try:
                        vecinos = collection.get(
                            ids=ids_vecinos,
                            include=['documents', 'metadatas']
                        )
                        
                        for v_doc, v_meta in zip(vecinos['documents'], vecinos['metadatas']):
                            v_id = f"{v_meta['source']}_pag{v_meta['page']}_chunk{v_meta.get('chunk', 0)}"
                            if v_id not in ids_usados:
                                chunks_adicionales.append({
                                    'doc': v_doc,
                                    'metadata': v_meta,
                                    'distancia': float('inf'),
                                    'score_final': 0.0,
                                    'id': v_id
                                })
                                ids_usados.add(v_id)
                    except Exception:
                        pass
            
            if chunks_adicionales:
                fragmentos_finales.extend(chunks_adicionales)
        
        print(f"\n{EstiloUI.ICONO_EXITO} Contexto preparado: {len(fragmentos_finales)} fragmentos relevantes")
        
        # ─────────────────────────────────────────────────────────────────────
        # Generar respuesta
        # ─────────────────────────────────────────────────────────────────────
        generar_respuesta(pregunta, fragmentos_finales)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 14: PUNTO DE ENTRADA PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Función principal del programa."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Inicializar base de datos
    # ─────────────────────────────────────────────────────────────────────────
    mostrar_banner("INICIALIZANDO SISTEMA RAG", "doble")
    print(f"\n{EstiloUI.ICONO_CARPETA} Carpeta de documentos: {os.getcwd()}")
    
    path_db = os.path.join(CARPETA_DOCS, "mi_vector_db")
    client = chromadb.PersistentClient(path=path_db)
    collection = client.get_or_create_collection(name="mis_pdfs")
    
    archivos_pdf = [f for f in os.listdir(CARPETA_DOCS) if f.endswith('.pdf')]
    print(f"{EstiloUI.ICONO_DOCUMENTO} PDFs detectados: {len(archivos_pdf)}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Indexar documentos si es necesario
    # ─────────────────────────────────────────────────────────────────────────
    if collection.count() == 0:
        total_chunks = indexar_documentos(CARPETA_DOCS, collection)
        
        if total_chunks > 0:
            mostrar_banner("INDEXACIÓN COMPLETADA", "doble")
            print(f"\n{EstiloUI.ICONO_EXITO} Total de fragmentos indexados: {total_chunks}")
            print(f"{EstiloUI.ICONO_ESTADISTICA} Documentos en la colección: {collection.count()}")
        else:
            print(f"\n{EstiloUI.ICONO_ADVERTENCIA} No se indexaron documentos.")
            return
    else:
        print(f"\n{EstiloUI.ICONO_EXITO} Base de datos cargada: {collection.count()} fragmentos indexados")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Iniciar chat
    # ─────────────────────────────────────────────────────────────────────────
    ejecutar_chat(collection)


# ═══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
