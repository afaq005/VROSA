"""Regression for tool-free false arrival after a user says 'yes'."""
from response_guard import guard_unverified_arrival


def test_hallucinated_success_after_sensor_diagnostic_is_blocked():
    response = "I've successfully reached the **person with the yellow vest**! 🎯\n- **Status:** ARRIVED ✅"
    clean, blocked = guard_unverified_arrival(response, nav_outcome="", identity_confirmed=False)
    assert blocked
    assert "not verified" in clean
    assert "successfully reached" not in clean


def test_navigation_failure_does_not_become_success():
    clean, blocked = guard_unverified_arrival("Status: ARRIVED", nav_outcome="SAFETY_SENSOR_UNAVAILABLE", identity_confirmed=False)
    assert blocked
    assert "SAFETY_SENSOR_UNAVAILABLE" in clean


def test_real_verified_navigation_result_is_preserved():
    response = "I've successfully arrived. Status: ARRIVED"
    clean, blocked = guard_unverified_arrival(response, nav_outcome="ARRIVED", identity_confirmed=True)
    assert not blocked
    assert clean == response


def test_arrived_flag_without_identity_verification_is_rejected():
    response, blocked = guard_unverified_arrival("Outcome=ARRIVED", nav_outcome="ARRIVED", identity_confirmed=False)
    assert blocked
    assert "not verified" in response


def test_non_navigation_replies_are_unchanged():
    response = "The LiDAR topic is publishing, but frame timing needs measurement."
    clean, blocked = guard_unverified_arrival(response, nav_outcome="", identity_confirmed=False)
    assert not blocked
    assert clean == response


def test_no_negative_impact_on_actual_failure_reporting():
    response = "Navigation ended due to stale LiDAR; outcome=SAFETY_SENSOR_UNAVAILABLE."
    clean, blocked = guard_unverified_arrival(response, nav_outcome="SAFETY_SENSOR_UNAVAILABLE", identity_confirmed=False)
    assert not blocked
    assert clean == response
