import os
import json
import re
import time
import html
import unicodedata
import requests


# CONFIGURACIÓN

DATASET_DIR = r"C:\Users\cjant\Downloads\Seed dataset\Seed dataset"

OUTPUT_FOLDER = "config"

CACHE_FILE = "gbif_cache.json"

MAX_RETRIES = 3

RETRY_BASE_DELAY = 2

REQUEST_PAUSE = 1.0

MIN_CONFIDENCE = 90

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}


# SESSION HTTP

session = requests.Session()

session.headers.update({
    "User-Agent": "BioCLIP-Seed-Dataset-Configurator/1.0"
})


# NORMALIZACIÓN DE TEXTO

def normalize_text(text: str) -> str:
    """
    Normaliza texto:
    - Unicode
    - espacios no separables
    - caracteres invisibles
    - espacios múltiples
    """

    if not text:
        return ""

    text = str(text)

    # Normalización Unicode
    text = unicodedata.normalize("NFKC", text)

    # Espacios no separables y caracteres similares
    text = text.replace("\u00A0", " ")
    text = text.replace("\u2007", " ")
    text = text.replace("\u202F", " ")

    # Eliminar caracteres invisibles / zero-width
    invisible_chars = [
        "\u200B",
        "\u200C",
        "\u200D",
        "\u2060",
        "\uFEFF"
    ]

    for char in invisible_chars:
        text = text.replace(char, "")

    # Eliminar caracteres de control
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Cc"
    )

    # Eliminar espacios repetidos
    text = " ".join(text.split())

    return text.strip()


# DETECCIÓN DE HTML / CARACTERES EXTRAÑOS

HTML_ENTITY_PATTERN = re.compile(
    r"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9A-Fa-f]+);"
)


def contains_html_entity(text: str) -> bool:
    return bool(HTML_ENTITY_PATTERN.search(text))


def contains_non_latin_script(text: str) -> bool:
    """
    Detecta escrituras que no corresponden al alfabeto latino,
    por ejemplo:
    - Cirílico
    - Chino
    - Japonés
    - Coreano
    - Árabe
    - Hebreo
    - Devanagari
    - Tailandés
    """

    blocked_keywords = [
        "CYRILLIC",
        "CJK",
        "HIRAGANA",
        "KATAKANA",
        "HANGUL",
        "ARABIC",
        "HEBREW",
        "DEVANAGARI",
        "THAI",
        "GEORGIAN",
        "ARMENIAN"
    ]

    for char in text:
        name = unicodedata.name(char, "")

        for keyword in blocked_keywords:
            if keyword in name:
                return True

    return False


def clean_common_name(raw_name: str) -> str:
    """
    Limpia y valida un nombre común.
    Si presenta HTML, scripts no latinos o caracteres
    de reemplazo, devuelve una cadena vacía.
    """

    if not raw_name:
        return ""

    original = str(raw_name)

    # Si GBIF devuelve una entidad HTML, la consideramos sospechosa
    if contains_html_entity(original):
        return ""

    # Decodificar HTML por seguridad
    cleaned = html.unescape(original)

    # Normalizar
    cleaned = normalize_text(cleaned)

    if not cleaned:
        return ""

    # Carácter de reemplazo Unicode
    if "\ufffd" in cleaned:
        return ""

    # Detectar idiomas/escrituras no latinas
    if contains_non_latin_script(cleaned):
        return ""

    return cleaned


# VALIDACIÓN DEL NOMBRE COMÚN EN INGLÉS

def get_english_common_name(results):
    """
    Busca exclusivamente nombres cuyo idioma sea inglés.
    Nunca utiliza como fallback un nombre de otro idioma.
    """

    for item in results:

        language = normalize_text(
            item.get("language", "")
        ).lower()

        if language != "en":
            continue

        raw_name = item.get("vernacularName", "")

        common_name = clean_common_name(raw_name)

        if common_name:
            return common_name

    return ""


# PETICIONES HTTP CON REINTENTOS

RETRY_STATUS_CODES = {
    408,
    425,
    429,
    500,
    502,
    503,
    504
}


def make_request(url: str, params=None):
    """
    Realiza una petición GET con:
    - timeout
    - reintentos automáticos
    - backoff
    - manejo de HTTP 429
    - pausa entre solicitudes
    """

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                url,
                params=params,
                timeout=10
            )

            # Error temporal
            if response.status_code in RETRY_STATUS_CODES:

                if attempt < MAX_RETRIES:

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    if retry_after:

                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = RETRY_BASE_DELAY * attempt

                    else:
                        delay = RETRY_BASE_DELAY * attempt

                    print(
                        f"  Error HTTP {response.status_code}. "
                        f"Reintentando en {delay:.1f}s..."
                    )

                    time.sleep(delay)

                    continue

                print(
                    f"  Error HTTP {response.status_code} "
                    f"después de {MAX_RETRIES} intentos."
                )

                return None

            response.raise_for_status()

            # Pausa entre solicitudes exitosas
            time.sleep(REQUEST_PAUSE)

            return response

        except requests.RequestException as error:

            if attempt < MAX_RETRIES:

                delay = RETRY_BASE_DELAY * attempt

                print(
                    f"  Error de conexión. "
                    f"Reintento {attempt}/{MAX_RETRIES - 1} "
                    f"en {delay}s..."
                )

                time.sleep(delay)

            else:

                print(
                    f"  Solicitud fallida después de "
                    f"{MAX_RETRIES} intentos: {error}"
                )

    return None


# CACHÉ GBIF

def load_cache(cache_path: str) -> dict:

    if not os.path.exists(cache_path):
        return {}

    try:

        with open(
            cache_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError
    ):

        print(
            " No se pudo leer la caché. "
            "Se iniciará una nueva."
        )

        return {}


def save_cache(cache_path: str, cache: dict):

    with open(
        cache_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cache,
            file,
            indent=2,
            ensure_ascii=False
        )


# CONSULTA DE GBIF

def get_species_from_gbif(
    scientific_name: str,
    cache: dict
):
    """
    Consulta GBIF para identificar la especie.

    Devuelve información taxonómica y el nombre común
    exclusivamente en inglés.
    """

    cache_key = normalize_text(
        scientific_name
    ).lower()

    # Revisar caché

    if cache_key in cache:

        return cache[cache_key]

    # Step 1: species match

    species_url = (
        "https://api.gbif.org/v1/species/match"
    )

    response = make_request(
        species_url,
        params={
            "name": scientific_name
        }
    )

    if response is None:

        result = {
            "common_name": scientific_name,
            "usage_key": None,
            "accepted_usage_key": None,
            "canonical_name": scientific_name,
            "gbif_scientific_name": None,
            "confidence": 0,
            "match_type": "ERROR",
            "rank": None,
            "status": None,
            "needs_review": True,
            "review_reason": "GBIF API no disponible"
        }

        cache[cache_key] = result

        return result

    try:

        data = response.json()

    except ValueError:

        result = {
            "common_name": scientific_name,
            "usage_key": None,
            "accepted_usage_key": None,
            "canonical_name": scientific_name,
            "gbif_scientific_name": None,
            "confidence": 0,
            "match_type": "INVALID_JSON",
            "rank": None,
            "status": None,
            "needs_review": True,
            "review_reason": "Respuesta JSON inválida"
        }

        cache[cache_key] = result

        return result

    usage_key = data.get("usageKey")

    accepted_usage_key = data.get(
        "acceptedUsageKey"
    )

    canonical_name = normalize_text(
        data.get(
            "canonicalName",
            scientific_name
        )
    )

    gbif_scientific_name = normalize_text(
        data.get(
            "scientificName",
            scientific_name
        )
    )

    confidence = data.get(
        "confidence",
        0
    )

    match_type = normalize_text(
        data.get(
            "matchType",
            "NONE"
        )
    ).upper()

    rank = normalize_text(
        data.get(
            "rank",
            ""
        )
    ).upper()

    status = normalize_text(
        data.get(
            "status",
            ""
        )
    ).upper()

    # Validación de coincidencia

    needs_review = False
    review_reason = ""

    if not usage_key:

        needs_review = True
        review_reason = "GBIF no encontró una coincidencia"

    elif confidence < MIN_CONFIDENCE:

        needs_review = True
        review_reason = (
            f"Confianza GBIF baja ({confidence})"
        )

    elif match_type != "EXACT":

        needs_review = True
        review_reason = (
            f"Tipo de coincidencia: {match_type}"
        )

    elif rank != "SPECIES":

        needs_review = True
        review_reason = (
            f"GBIF identificó el taxón como {rank}"
        )

    # Seleccionar clave para vernacular names

    vernacular_key = (
        accepted_usage_key
        or usage_key
    )

    common_name = ""

    # Step 2: nombres comunes

    if vernacular_key:

        names_url = (
            f"https://api.gbif.org/v1/species/"
            f"{vernacular_key}/vernacularNames"
        )

        names_response = make_request(
            names_url
        )

        if names_response is not None:

            try:

                names_data = names_response.json()

                results = names_data.get(
                    "results",
                    []
                )

                common_name = get_english_common_name(
                    results
                )

            except ValueError:

                pass

    # Fallback seguro

    if not common_name:

        common_name = scientific_name

        if not review_reason:

            review_reason = (
                "No se encontró nombre común válido en inglés"
            )

            needs_review = True

    result = {

        "common_name": common_name,

        "usage_key": usage_key,

        "accepted_usage_key": accepted_usage_key,

        "canonical_name": canonical_name,

        "gbif_scientific_name": gbif_scientific_name,

        "confidence": confidence,

        "match_type": match_type,

        "rank": rank,

        "status": status,

        "needs_review": needs_review,

        "review_reason": review_reason

    }

    cache[cache_key] = result

    return result


# CONTEO DE IMÁGENES

def count_images(folder_path: str) -> int:

    count = 0

    try:

        for file_name in os.listdir(folder_path):

            file_path = os.path.join(
                folder_path,
                file_name
            )

            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(
                file_name
            )[1].lower()

            if extension in IMAGE_EXTENSIONS:
                count += 1

    except OSError:

        return 0

    return count


# RUTA ÚNICA DE SALIDA

def get_unique_config_path(
    base_folder="config",
    base_name="species_config"
):

    os.makedirs(
        base_folder,
        exist_ok=True
    )

    candidate_path = os.path.join(
        base_folder,
        f"{base_name}.json"
    )

    counter = 2

    while os.path.exists(candidate_path):

        candidate_path = os.path.join(
            base_folder,
            f"{base_name}_{counter}.json"
        )

        counter += 1

    return candidate_path


# CONSTRUCCIÓN DEL JSON

def build_config_from_folders(
    dataset_path: str,
    output_folder: str = OUTPUT_FOLDER
):

    # Validar dataset

    if not os.path.exists(dataset_path):

        print(
            f"La ruta no existe:\n"
            f"{dataset_path}"
        )

        return

    if not os.path.isdir(dataset_path):

        print(
            f"La ruta no corresponde a una carpeta:\n"
            f"{dataset_path}"
        )

        return

    print(
        f"Ruta del dataset encontrada:\n"
        f"{dataset_path}\n"
    )

    # Buscar carpetas

    folders = sorted(
        [
            folder
            for folder in os.listdir(dataset_path)
            if os.path.isdir(
                os.path.join(
                    dataset_path,
                    folder
                )
            )
        ]
    )

    if not folders:

        print(
            "No se encontraron carpetas "
            "de especies dentro del dataset."
        )

        return

    print(
        f" Carpetas encontradas: "
        f"{len(folders)}\n"
    )

    # Preparar caché

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    cache_path = os.path.join(
        output_folder,
        CACHE_FILE
    )

    cache = load_cache(cache_path)

    species_list = []

    species_by_taxon = {}

    review_list = []

    duplicate_list = []

    folders_without_images = []

    # Procesar carpetas

    for idx, folder_name in enumerate(
        folders,
        start=1
    ):

        folder_path = os.path.join(
            dataset_path,
            folder_name
        )

        # Normalizar nombre científico

        scientific_name = folder_name.replace(
            "_",
            " "
        )

        scientific_name = normalize_text(
            scientific_name
        )

        # Contar imágenes

        image_count = count_images(
            folder_path
        )

        if image_count == 0:

            folders_without_images.append(
                folder_name
            )

        # Consultar GBIF

        print(
            f"[{idx}/{len(folders)}] "
            f"{scientific_name}"
        )

        gbif_info = get_species_from_gbif(
            scientific_name,
            cache
        )

        common_name = gbif_info[
            "common_name"
        ]

        # Determinar clave taxonómica

        taxon_key = (
            gbif_info["accepted_usage_key"]
            or gbif_info["usage_key"]
            or normalize_text(
                gbif_info["canonical_name"]
            ).lower()
        )

        # Detectar duplicados

        duplicate_of = None

        if taxon_key in species_by_taxon:

            duplicate_of = species_by_taxon[
                taxon_key
            ]

            duplicate_list.append({

                "folder_name": folder_name,

                "scientific_name": scientific_name,

                "duplicate_of": duplicate_of,

                "reason": (
                    "Misma clave taxonómica de GBIF "
                    "o mismo canonicalName."
                )

            })

        else:

            species_by_taxon[
                taxon_key
            ] = f"sp_{idx:04d}"

        # Registrar revisión

        if gbif_info["needs_review"]:

            review_list.append({

                "id": f"sp_{idx:04d}",

                "scientific_name": scientific_name,

                "common_name": common_name,

                "reason": gbif_info[
                    "review_reason"
                ]

            })

        # Crear registro

        species_entry = {

            "id": f"sp_{idx:04d}",

            "scientific_name": scientific_name,

            "common_name": common_name,

            "folder_name": folder_name,

            "image_count": image_count,

            "gbif_usage_key": gbif_info[
                "usage_key"
            ],

            "gbif_accepted_usage_key": gbif_info[
                "accepted_usage_key"
            ],

            "gbif_canonical_name": gbif_info[
                "canonical_name"
            ],

            "gbif_scientific_name": gbif_info[
                "gbif_scientific_name"
            ],

            "gbif_confidence": gbif_info[
                "confidence"
            ],

            "gbif_match_type": gbif_info[
                "match_type"
            ],

            "gbif_rank": gbif_info[
                "rank"
            ],

            "gbif_status": gbif_info[
                "status"
            ],

            "needs_review": gbif_info[
                "needs_review"
            ]

        }

        if duplicate_of:

            species_entry[
                "possible_duplicate_of"
            ] = duplicate_of

        species_list.append(
            species_entry
        )

        print(
            f"   → {common_name}"
        )

        if gbif_info["needs_review"]:

            print(
                f" Revisar: "
                f"{gbif_info['review_reason']}"
            )

        if duplicate_of:

            print(
                f" Posible duplicado de "
                f"{duplicate_of}"
            )

        if image_count == 0:

            print(
                "  La carpeta no contiene imágenes."
            )

        print()

    # Guardar caché

    save_cache(
        cache_path,
        cache
    )

    # Crear configuración

    config_data = {

        "templates": [

            "a close-up photograph of the seed of "
            "{scientific_name}, a plant species"

        ],

        "dataset": {

            "dataset_path": os.path.abspath(
                dataset_path
            ),

            "total_folders": len(folders),

            "total_species_records": len(
                species_list
            )

        },

        "summary": {

            "species_requiring_review": len(
                review_list
            ),

            "possible_duplicates": len(
                duplicate_list
            ),

            "folders_without_images": len(
                folders_without_images
            )

        },

        "species": species_list,

        "quality_control": {

            "species_requiring_review": review_list,

            "possible_duplicates": duplicate_list,

            "folders_without_images": (
                folders_without_images
            )

        }

    }

    # Ruta de salida

    output_config_path = get_unique_config_path(
        base_folder=output_folder,
        base_name="species_config"
    )

    # Guardar JSON

    with open(
        output_config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config_data,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        "✅ CONFIGURACIÓN GENERADA CORRECTAMENTE"
    )

    print(
        "=================================================="
    )

    print(
        f"Dataset:\n"
        f"{os.path.abspath(dataset_path)}"
    )

    print(
        f"\nCarpetas encontradas: "
        f"{len(folders)}"
    )

    print(
        f"Registros generados: "
        f"{len(species_list)}"
    )

    print(
        f"Especies para revisar: "
        f"{len(review_list)}"
    )

    print(
        f"Posibles duplicados: "
        f"{len(duplicate_list)}"
    )

    print(
        f"Carpetas sin imágenes: "
        f"{len(folders_without_images)}"
    )

    print(
        f"\nArchivo generado:\n"
        f"{output_config_path}"
    )

    print(
        f"\nCaché GBIF:\n"
        f"{cache_path}"
    )

if __name__ == "__main__":

    build_config_from_folders(
        DATASET_DIR
    )