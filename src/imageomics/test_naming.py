from . import naming


def test_strip_html():
    raw = "Zygodon <i>viridissimus</i> viridissimus"
    expected = "Zygodon viridissimus viridissimus"
    assert naming.strip_html(raw) == expected


def test_strip_html_recursive():
    raw = "<i>Zygodon <i>viridissimus</i></i> viridissimus"
    expected = "Zygodon viridissimus viridissimus"
    assert naming.strip_html(raw) == expected


def test_clean_name_good_names():
    names = ["Gigartina papillata", "Uncinia hookeri", "Ostrea"]
    for name in names:
        assert naming.clean_name(name) == name.lower()


def test_clean_name_parens():
    raw = "Atrypanius cinerascens (Bates 1864)"
    expected = "atrypanius cinerascens"
    assert naming.clean_name(raw) == expected


def test_clean_name_comma_in_parens():
    raw = "Xymene huttoni (Murdoch, 1900)"
    expected = "xymene huttoni"
    assert naming.clean_name(raw) == expected


def test_clean_name_extra_after_parens():
    raw = "Huperzia dacrydioides (Baker) Pic.Serm."
    expected = "huperzia dacrydioides"
    assert naming.clean_name(raw) == expected


def test_clean_name_subspecies():
    raw = "Lagophylla ramosissima ssp. ramosissima"
    expected = "lagophylla ramosissima"
    assert naming.clean_name(raw) == expected


def test_clean_name_subspecies2():
    raw = "Lagocheirus araneiformis subsp. parvulus Casey 1913"
    expected = "lagocheirus araneiformis"
    assert naming.clean_name(raw) == expected


def test_clean_name_name_year():
    raw = "Megacyllene asteca Chevrolat 1860"
    expected = "megacyllene asteca"
    assert naming.clean_name(raw) == expected


def test_clean_name_name_comma_year():
    raw = "Partula subangulata Pease, 1870"
    expected = "partula subangulata"
    assert naming.clean_name(raw) == expected


def test_clean_name_variant():
    raw = "Echinocereus rigidissimus var. rubrispinus"
    expected = "echinocereus rigidissimus"
    assert naming.clean_name(raw) == expected


def test_clean_name_three_names():
    raw = "Collotheca orchidacea Meksuwan, Pholpunthin & Segers, 2013"
    expected = "collotheca orchidacea"
    assert naming.clean_name(raw) == expected


def test_clean_name_ampersand():
    raw = "Agosia yarrowi Jordan & Evermann"
    expected = "agosia yarrowi"
    assert naming.clean_name(raw) == expected


def test_clean_name_linebreak():
    raw = "Pellaea boivinii <br > var. boivinii"
    expected = "pellaea boivinii"
    assert naming.clean_name(raw) == expected


def test_clean_name_initials():
    raw = "Stylidium spinulosum r.br."
    expected = "stylidium spinulosum"
    assert naming.clean_name(raw) == expected


def test_find_initial_name():
    taxon = naming.Taxon("k", "p", "", "o", "f", "g", "s")
    expected = ("o", "order")
    assert naming.find_initial_name(taxon) == expected


def test_find_initial_name_no_genus():
    taxon = naming.Taxon("k", "p", "", "o", "f", "", "s")
    expected = ("o", "order")
    assert naming.find_initial_name(taxon) == expected


def test_find_initial_name_no_genus2():
    taxon = naming.Taxon("k", "p", "", "", "f", "", "s")
    expected = ("f", "family")
    assert naming.find_initial_name(taxon) == expected


def test_find_intial_name_two_holes():
    taxon = naming.Taxon("k", "", "c", "", "f", "g", "s")
    expected = ("f", "family")
    assert naming.find_initial_name(taxon) == expected


def test_find_intial_name_failed_order():
    taxon = naming.Taxon("k", "p", "", "o", "f", "g", "s")
    expected = ("f", "family")
    assert naming.find_initial_name(taxon, failed="order") == expected


def test_find_intial_name_failed_family():
    taxon = naming.Taxon("k", "p", "", "o", "f", "g", "s")
    expected = (("g", "s"), "scientific")
    assert naming.find_initial_name(taxon, failed="family") == expected


def test_find_intial_name_failed_scientific():
    taxon = naming.Taxon("k", "p", "", "o", "f", "g", "s")
    expected = (None, None)
    assert naming.find_initial_name(taxon, failed="scientific") == expected


def test_taxon_fills_genus_from_species_with_initials():
    species = "Morinda citrifolia L."
    expected = naming.Taxon("", "", "", "", "", "Morinda", "citrifolia")
    assert naming.Taxon("", "", "", "", "", "", species) == expected


def test_taxon_fills_genus_from_species_with_two_initials():
    species = "Stylidium spinulosum r.br."
    expected = naming.Taxon("", "", "", "", "", "stylidium", "spinulosum")
    assert naming.Taxon("", "", "", "", "", "", species) == expected


def test_taxon_fills_genus_from_species_with_biologist_name():
    species = "Drosera aliciae r.hamet"
    expected = naming.Taxon("", "", "", "", "", "drosera", "aliciae")
    assert naming.Taxon("", "", "", "", "", "", species) == expected


def test_taxon_removes_duplicate_species():
    species = "geospiza fuliginosa fuliginosa"
    expected = naming.Taxon("", "", "", "", "", "geospiza", "fuliginosa")
    assert naming.Taxon("", "", "", "", "", "", species) == expected


def test_taxon_removes_duplicate_species2():
    species = "geospiza fuliginosa fuliginosa fuliginosa"
    expected = naming.Taxon("", "", "", "", "", "geospiza", "fuliginosa")
    assert naming.Taxon("", "", "", "", "", "", species) == expected
