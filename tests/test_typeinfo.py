from righttyper.typeinfo import TypeInfo


def test_is_typevar():
    assert not TypeInfo.from_type(int).is_typevar()
    assert not TypeInfo.from_set(set((
        TypeInfo("", "list", args=(
            TypeInfo.from_type(int),
        )),
        TypeInfo.from_type(int),
        TypeInfo.from_type(bool),
    ))).is_typevar()

    assert TypeInfo.from_set(
        set((
            TypeInfo.from_type(int),
            TypeInfo.from_type(bool),
        )),
        typevar_index=1
    ).is_typevar()

    assert TypeInfo("", "list", args=(
        TypeInfo.from_set(
            set((
                TypeInfo.from_type(int),
                TypeInfo.from_type(bool),
            )),
            typevar_index=1
        ),
    )).is_typevar()
