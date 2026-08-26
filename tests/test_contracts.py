from representation_reliability.contracts import RepresentationSite, Sample, TokenSelection

def test_sample_counterfactual_fields_optional() -> None:
    s = Sample(sample_id="s1", prompt="A", target_label=1, task_name="unit")
    assert s.counterfactual_id is None

def test_representation_site_has_resolved_token() -> None:
    tok = TokenSelection(strategy="explicit", index=3, token_id=42, token_text="foo")
    site = RepresentationSite(site="resid_post", layer=2, token=tok)
    assert site.token.token_text == "foo"
    assert site.layer == 2
