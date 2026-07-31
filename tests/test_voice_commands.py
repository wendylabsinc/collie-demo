from voice.commands import parse_voice_command


def test_find_command_accepts_only_allowlisted_fruit_phrases() -> None:
    assert parse_voice_command("Find the apple").target == "apple"
    assert parse_voice_command("woof, find banana!").target == "banana"
    assert parse_voice_command("Hey Woof, please find the pear.").target == "pear"
    assert parse_voice_command("please find the apple please").target == "apple"


def test_bare_fruit_command_and_pear_homophone_are_allowlisted() -> None:
    assert parse_voice_command("apple").target == "apple"
    assert parse_voice_command("Banana!").target == "banana"
    assert parse_voice_command("pear.").target == "pear"
    assert parse_voice_command("pair").target == "pear"
    assert parse_voice_command("find pair").target == "pear"


def test_find_command_rejects_free_form_or_unsupported_targets() -> None:
    assert parse_voice_command("Find the orange") is None
    assert parse_voice_command("walk forward and find the apple") is None
    assert parse_voice_command("find the apple then keep walking") is None
    assert parse_voice_command("") is None


def test_stop_command_is_deterministic() -> None:
    assert parse_voice_command("Stop").kind == "stop"
    assert parse_voice_command("Stop now!").kind == "stop"
    assert parse_voice_command("Woof stop").kind == "stop"
    assert parse_voice_command("Hey Woof, abort mission!").kind == "stop"
    assert parse_voice_command("please stop walking") is None
