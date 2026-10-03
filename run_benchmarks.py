"""
Test Runner & Benchmark Suite for Smart Study Notes Generator
=============================================================
Runs at least 3 diverse domain paragraphs through the pre-trained Generative AI model,
measures metrics (original word count, summary word count, % reduction),
records the outputs, and evaluates relevance and coherence.
"""

import os
import json
import time
from smart_study_notes import SmartStudyNotesGenerator, print_formatted_results

# Define 3 rich, diverse test paragraphs
TEST_PARAGRAPHS = [
    {
        "id": "Test_01",
        "domain": "Artificial Intelligence & Healthcare",
        "topic": "AI-Driven Diagnostics and Clinical Decision Support",
        "text": (
            "Artificial intelligence in healthcare is revolutionizing how medical professionals diagnose and "
            "treat complex diseases. Machine learning algorithms analyze intricate medical imaging datasets, "
            "such as MRI scans and radiological X-rays, with remarkable precision, often detecting subtle "
            "malignant lesions that could easily escape human observation. Furthermore, AI-powered predictive "
            "models assist hospitals in forecasting patient admission surges and optimizing vital resource "
            "allocation across intensive care units. Continuous remote patient monitoring through wearable biosensors "
            "enables timely clinical interventions for chronic condition management. Although these transformative "
            "technological advancements promise to elevate patient survival rates and lower administrative expenses, "
            "seamlessly integrating AI into clinical workflows necessitates addressing critical hurdles regarding "
            "patient data confidentiality, algorithmic bias, regulatory compliance, and physician oversight."
        )
    },
    {
        "id": "Test_02",
        "domain": "Environmental Science & Energy Systems",
        "topic": "Renewable Energy Transition and Grid Modernization",
        "text": (
            "The global transition toward renewable energy is accelerating rapidly in response to escalating climate "
            "change and the imperative to drastically reduce greenhouse gas emissions. Solar photovoltaic panels "
            "and wind turbines are now generating electricity at costs highly competitive with conventional fossil "
            "fuels, driving unprecedented investments across industrialized and developing economies alike. However, "
            "the inherent intermittency of wind and sunlight poses formidable engineering challenges for national "
            "electrical grids, requiring massive deployment of advanced grid-scale battery storage, pumped hydroelectric "
            "systems, and modernized high-voltage direct-current transmission networks. Transitioning away from "
            "legacy coal and gas infrastructure also carries significant socioeconomic repercussions, compelling "
            "policymakers to devise just transition frameworks that provide green workforce reskilling, energy "
            "equity, and economic diversification for displaced fossil fuel communities."
        )
    },
    {
        "id": "Test_03",
        "domain": "Astrophysics & Space Exploration",
        "topic": "James Webb Space Telescope & Early Universe Discoveries",
        "text": (
            "The James Webb Space Telescope represents an unprecedented technological leap in observational "
            "astronomy, offering humanity its deepest and most detailed view into the primordial cosmos. Equipped "
            "with high-resolution infrared detectors and an enormous gold-coated beryllium mirror, the observatory "
            "can pierce through dense interstellar dust clouds to observe the very first galaxies that coalesced just "
            "a few hundred million years following the Big Bang. In addition to probing cosmic origins, the telescope "
            "performs sophisticated atmospheric spectroscopy on exoplanets orbiting distant stellar systems, searching "
            "for molecular biosignatures such as water vapor, methane, and carbon dioxide. Operating at the Second "
            "Lagrange Point nearly one million miles from Earth, this flagship international mission is systematically "
            "dismantling long-held astrophysical assumptions while revolutionizing our understanding of stellar "
            "genesis and planetary habitability."
        )
    }
]


def run_benchmarks():
    print("=" * 78)
    print("      SMART STUDY NOTES GENERATOR - AUTOMATED BENCHMARK & EVALUATION")
    print("=" * 78)
    print(f"Loaded {len(TEST_PARAGRAPHS)} distinct academic benchmark test cases.\n")

    generator = SmartStudyNotesGenerator()
    results = []

    for idx, test_case in enumerate(TEST_PARAGRAPHS, 1):
        print(f"\n[{idx}/{len(TEST_PARAGRAPHS)}] Running Test: {test_case['topic']} ({test_case['domain']})...")
        res = generator.generate_notes(test_case["text"])
        res["id"] = test_case["id"]
        res["domain"] = test_case["domain"]
        res["topic"] = test_case["topic"]
        results.append(res)
        print_formatted_results(res, title=f"Test Case {idx}: {test_case['topic']}")

    # Print Summary Table
    print("\n" + "=" * 78)
    print("                        BENCHMARK SUMMARY TABLE")
    print("=" * 78)
    print(f"{'Test ID':<9} | {'Domain':<26} | {'Original':<9} | {'Summary':<8} | {'Reduced':<8} | {'% Reduction':<11}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['id']:<9} | "
            f"{r['domain'][:26]:<26} | "
            f"{r['original_word_count']:<9} | "
            f"{r['summary_word_count']:<8} | "
            f"{r['words_reduced']:<8} | "
            f"{r['percentage_reduction']:<11}%"
        )
    print("=" * 78)

    # Save results to JSON file for reporting
    json_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full test records saved to '{json_path}'.")

    return results


if __name__ == "__main__":
    run_benchmarks()
