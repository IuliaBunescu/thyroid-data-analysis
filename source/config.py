import plotly.express as px
import plotly.io as pio

PINK_RED_PALETTE = [
    "#E63946",  # red-pink
    "#F4A1A8",  # light rose
    "#FFD6DA",  # blush
    "#C2185B",  # magenta
    "#5A1E1E",  # dark red-brown
    "#FFB5A7",  # peach
    "#F8C6CC",  # light pink
]

DISCRETE_COLOR_PALETTE = px.colors.colorbrewer.RdBu
CUSTOM_DISCRETE_2VAR_COLOR_PALETTE = [
    PINK_RED_PALETTE[0],
    "#DEF2FF",
]  # red-pink and blue
CONTINUOUS_COLOR_SCALE = "RdBu_r"

pio.templates["thyroid_theme"] = pio.templates["plotly"].update(
    {
        "layout": {
            "colorway": PINK_RED_PALETTE,
            "title": {"font": {"size": 22, "color": "#5A1E1E"}},
        }
    }
)

pio.templates.default = "thyroid_theme"

TITLE_FONT_SIZE = 22
AXIS_TITLE_FONT_SIZE = 18
AXIS_TICK_FONT_SIZE = 16


COLUMN_DESCRIPTIONS = {
    "age": {
        "short": "Patient's age in years.",
        "link": "https://en.wikipedia.org/wiki/Ageing",
    },
    "sex": {
        "short": "Biological sex of the patient.",
        "link": "https://en.wikipedia.org/wiki/Sex",
    },
    "on_thyroxine": {
        "short": "Whether the patient is taking thyroxine.",
        "long": "Indicates if the patient is undergoing treatment with thyroxine, a synthetic form of the thyroid hormone used to treat hypothyroidism.",
        "link": "https://en.wikipedia.org/wiki/Levothyroxine",
    },
    "query_on_thyroxine": {
        "short": "Whether it's suspected the patient is on thyroxine.",
        "long": "Indicates if there's suspicion that the patient is taking thyroxine, based on clinical assessment or patient history.",
        "link": "https://en.wikipedia.org/wiki/Levothyroxine",
    },
    "on_antithyroid_medication": {
        "short": "Whether the patient is on antithyroid medication.",
        "long": "Indicates if the patient is being treated with medications that inhibit thyroid hormone production, commonly used for hyperthyroidism.",
        "link": "https://en.wikipedia.org/wiki/Antithyroid_agent",
    },
    "sick": {
        "short": "Whether the patient is currently sick.",
        "long": "Indicates if the patient is experiencing illness, which may include symptoms like fever, fatigue, or pain.",
        "link": "https://en.wikipedia.org/wiki/Sick",
    },
    "pregnant": {
        "short": "Whether the patient is pregnant.",
        "link": "https://en.wikipedia.org/wiki/Pregnancy",
    },
    "thyroid_surgery": {
        "short": "Whether the patient has had thyroid surgery.",
        "long": "Indicates if the patient has undergone surgical procedures involving the thyroid gland, such as thyroidectomy.",
        "link": "https://en.wikipedia.org/wiki/Thyroidectomy",
    },
    "I131_treatment": {
        "short": "Whether the patient has received Iodine-131 treatment.",
        "long": "Indicates if the patient has been treated with radioactive iodine (I-131), commonly used for conditions like hyperthyroidism or thyroid cancer.",
        "link": "https://en.wikipedia.org/wiki/Iodine-131",
    },
    "query_hypothyroid": {
        "short": "Whether the patient believes they have hypothyroidism.",
        "long": "Indicates if the patient believes they have hypothyroidism, a condition where the thyroid gland doesn't produce enough hormones.",
        "link": "https://en.wikipedia.org/wiki/Hypothyroidism",
    },
    "query_hyperthyroid": {
        "short": "Whether the patient believes they have hyperthyroidism.",
        "long": "Indicates if the patient believes they have hyperthyroidism, a condition where the thyroid gland produces too much hormone.",
        "link": "https://en.wikipedia.org/wiki/Hyperthyroidism",
    },
    "lithium": {
        "short": "Whether the patient is taking lithium.",
        "long": "Indicates if the patient is undergoing treatment with lithium, a medication commonly used for bipolar disorder.",
        "link": "https://en.wikipedia.org/wiki/Lithium_(medication)",
    },
    "goitre": {
        "short": "Presence of goitre (enlarged thyroid).",
        "long": "Indicates if the patient has a visible enlargement of the thyroid gland, which can be a sign of thyroid dysfunction.",
        "link": "https://en.wikipedia.org/wiki/Goitre",
    },
    "tumor": {
        "short": "Presence of a thyroid tumor.",
        "long": "Indicates if the patient has a growth or mass in the thyroid gland, which may be benign or malignant.",
        "link": "https://en.wikipedia.org/wiki/Thyroid_cancer",
    },
    "hypopituitary": {
        "short": "Whether the patient has hypopituitarism.",
        "long": "Indicates if the patient has hypopituitarism, a condition where the pituitary gland doesn't produce sufficient amounts of certain hormones.",
        "link": "https://en.wikipedia.org/wiki/Hypopituitarism",
    },
    "psych": {
        "short": "Whether the patient has psychiatric conditions.",
        "long": "Indicates if the patient has been diagnosed with mental health disorders, such as depression or anxiety.",
        "link": "https://en.wikipedia.org/wiki/Psychiatry",
    },
    "TSH_measured": {
        "short": "Whether TSH was measured.",
        "long": "Indicates if the patient's thyroid-stimulating hormone (TSH) levels were tested, which helps assess thyroid function.",
        "link": "https://en.wikipedia.org/wiki/Thyroid_stimulating_hormone",
    },
    "TSH": {
        "short": "TSH level (thyroid-stimulating hormone).",
        "long": "The concentration of thyroid-stimulating hormone in the blood, used to evaluate thyroid function.",
        "link": "https://en.wikipedia.org/wiki/Thyroid_stimulating_hormone",
    },
    "T3_measured": {
        "short": "Whether T3 was measured.",
        "long": "Indicates if the patient's triiodothyronine (T3) levels were tested, another key thyroid hormone.",
        "link": "https://en.wikipedia.org/wiki/Triiodothyronine",
    },
    "T3": {
        "short": "T3 level (triiodothyronine).",
        "long": "The concentration of triiodothyronine in the blood, important for regulating metabolism.",
        "link": "https://en.wikipedia.org/wiki/Triiodothyronine",
    },
    "TT4_measured": {
        "short": "Whether total T4 was measured.",
        "long": "Indicates if the patient's total thyroxine (T4) levels were tested, which reflects thyroid function.",
        "link": "https://en.wikipedia.org/wiki/Thyroxine",
    },
    "TT4": {
        "short": "Total T4 level (thyroxine).",
        "long": "The overall concentration of thyroxine in the blood, including both bound and free forms.",
        "link": "https://en.wikipedia.org/wiki/Thyroxine",
    },
    "T4U_measured": {
        "short": "Whether T4U was measured.",
        "long": "Indicates if the patient's thyroxine-binding globulin (TBG) levels were tested, which can affect thyroid hormone levels.",
        "link": "https://en.wikipedia.org/wiki/Thyroxine-binding_globulin",
    },
    "T4U": {
        "short": "T4U level (thyroxine-binding globulin).",
        "long": "The concentration of thyroxine-binding globulin in the blood, influencing the amount of free thyroid hormones.",
        "link": "https://en.wikipedia.org/wiki/Thyroxine-binding_globulin",
    },
    "FTI_measured": {
        "short": "Whether FTI was measured.",
        "long": "Indicates if the patient's free thyroxine index was tested, which helps assess thyroid function.",
        "link": "https://en.wikipedia.org/wiki/Free_thyroxine_index",
    },
    "FTI": {
        "short": "FTI level (free thyroxine index).",
        "long": "A calculated value used to evaluate thyroid function, considering both T4 and TBG levels.",
        "link": "https://en.wikipedia.org/wiki/Free_thyroxine_index",
    },
    "TBG_measured": {
        "short": "Whether TBG was measured.",
        "long": "Indicates if the patient's thyroxine-binding globulin levels were tested.",
        "link": "https://en.wikipedia.org/wiki/Thyroxine-binding_globulin",
    },
    "TBG": {
        "short": "TBG level (thyroxine-binding globulin).",
        "long": "The concentration of thyroxine-binding globulin in the blood.",
        "link": "https://en.wikipedia.org/wiki/Thyroxine-binding_globulin",
    },
    "referral_source": {
        "short": "Source that referred the patient.",
        "long": "Indicates the origin of the patient's referral for testing, such as a primary care physician or endocrinologist.",
        "link": "https://en.wikipedia.org/wiki/Referral",
    },
    "condition_primary": {
        "short": "Primary diagnostic code.",
        "long": "The main diagnostic category or group represented by the first character of the condition code (e.g., A = hypothyroidism).",
        "link": "https://en.wikipedia.org/wiki/Diagnosis",
    },
    "condition_secondary": {
        "short": "Secondary diagnostic subcode.",
        "long": "A subcategory or finer diagnostic detail represented by the second character of the condition code, if available.",
        "link": "https://en.wikipedia.org/wiki/Diagnosis#Classification",
    },
    "condition_primary": {
        "short": "Primary diagnosis letter.",
        "long": (
            "The first letter of the condition code, representing the main diagnostic category. "
            "For example, in 'AB', 'A' is the primary diagnostic condition."
        ),
    },
    "condition_secondary": {
        "short": "Secondary diagnosis letter (if present).",
        "long": (
            "The second letter of the condition code, representing an additional diagnostic comment "
            "or related condition. For example, in 'AB', 'B' is the secondary diagnostic condition."
        ),
    },
}
