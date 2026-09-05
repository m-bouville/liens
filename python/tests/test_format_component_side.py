"""format_component_side single-sources the '{total} =c0 +c1 +c2 ...' width
convention shared by the component-decomposition epoch lines (refinement, stage 2,
stage 1). This pins it BYTE-FOR-BYTE against the inline pattern it replaced, so the
refactor can't have changed a printed line, and a future width tweak stays in one
place. (train_lds's rollout/1step line is a different format and is not covered.)"""
from training._training_loop import format_component_side


def _inline(total, c):
    """The exact old inline pattern (=c0 +c1 +c2 ...), reproduced here as the oracle."""
    return (f"{total:7.4f} ={c[0]:7.4f}"
            + "".join(f" +{x:7.4f}" for x in c[1:]))


def test_matches_the_inline_pattern_byte_for_byte():
    for total, contribs in [
        (4.1476, [0.3825, 0.1822, 0.0561, 1.7841, 1.7426]),   # 5 active (refinement full)
        (0.6309, [0.1814, 0.0600, 0.0468]),                    # 3 active (ramp off)
        (3.7139, [0.3602, 0.3655, 0.0527, 1.4197, 1.5157]),    # val side
        (1.0, [0.0]),                                          # single component
    ]:
        assert format_component_side(total, contribs) == _inline(total, contribs)


def test_width_is_7_point_4f_and_plus_separated():
    out = format_component_side(2.0, [1.0, 0.5])
    assert out == " 2.0000 = 1.0000 + 0.5000"    # exact spacing/width contract
