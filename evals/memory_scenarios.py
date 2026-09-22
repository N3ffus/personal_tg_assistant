"""Long-term memory scenarios: several turns, then one question about them.

Every turn goes through the real bot. Unlike ``scenarios.py`` nothing here is
frozen: the facts reach Neo4j only if the model decided to remember them, and
the answer is grounded only if recall actually finds them again.

Every person, city, place, company, trip and date here is invented. Keep it
that way: nothing in these scenarios may describe a real user.

Fragments in the hard checks are compared case-insensitively. A tuple of
alternatives passes when any one of them occurs.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from src.domain.assistant.context import MAX_MESSAGES

# A Monday evening: «на выходных» then means the Saturday two days before.
MEMORY_NOW = datetime.fromisoformat("2041-06-03T19:00:00+03:00")

Alternatives = tuple[str, ...]


@dataclass(frozen=True)
class Turn:
    text: str
    # Days after MEMORY_NOW; the moment the user sends the message.
    day: float = 0
    # Another user of the same bot, for the namespace isolation scenario.
    user: Literal["owner", "stranger"] = "owner"
    # A different Telegram chat starts with an empty conversation history.
    chat: int = 1
    # Unrelated exchanges written into the chat history before this turn.
    filler: int = 0
    # Stop every client and open them again, as a redeployed backend does.
    restart: bool = False
    # Wipe this chat's history first, as /clear does.
    clear_history: bool = False


@dataclass(frozen=True)
class MemoryScenario:
    id: str
    turns: tuple[Turn, ...]
    expected: str
    # History the bot keeps per chat; lower it to push a fact out of the window.
    history_window: int = MAX_MESSAGES
    reply_contains: tuple[Alternatives, ...] = ()
    reply_excludes: tuple[str, ...] = ()
    # The final answer must have consulted long-term memory.
    recall_required: bool = False
    # The final turn's conversation history must not hold these: memory alone
    # can supply them.
    history_excludes: tuple[str, ...] = ()
    # Some earlier turn must have written each of these into memory.
    remembered: tuple[Alternatives, ...] = ()
    # No turn may write these into memory.
    never_remembered: tuple[str, ...] = ()
    # What the final turn's memory lookups must not return.
    lookup_excludes: tuple[str, ...] = ()
    # A direct memory search run after the dialog, bypassing the model.
    probe_query: str = ""
    probe_contains: tuple[Alternatives, ...] = ()
    probe_excludes: tuple[str, ...] = ()
    # (name fragment, maximum entity nodes whose name contains it).
    max_entities: tuple[str, int] | None = None
    # Ask the question of the shared PERSONA memory instead of a fresh namespace.
    persona: bool = False
    # The reply's list must hold exactly this many items.
    item_count: int | None = None
    # No list item may mention these: regexes matched at a word start, after
    # negated phrases («без грибов») are dropped from the item.
    items_exclude: tuple[str, ...] = ()
    # Each group must first appear in the reply after the previous one.
    reply_order: tuple[Alternatives, ...] = ()
    # (fragment, most occurrences the reply may hold).
    max_mentions: tuple[tuple[str, int], ...] = ()

    @property
    def question(self) -> Turn:
        return self.turns[-1]

    @property
    def setup(self) -> tuple[Turn, ...]:
        return self.turns[:-1]


@dataclass(frozen=True)
class Memory:
    """One fact of the persona, stored as the bot's remember_knowledge does."""

    text: str
    # When the user said it; memory keeps this as the fact's valid_from.
    stated: str


# A fictional user with enough history for questions that cross many memories.
# It is written once through the real adapter, in order, so Graphiti sees the
# later weight or employer as an update of the earlier one. Every persona
# scenario reads it; the ground truth in their expectations comes from here.
PERSONA = (
    Memory("Я живу в Леснограде", "2040-09-01"),
    Memory("Я инженер-гидролог", "2040-09-01"),
    Memory("Я работаю в компании «Пиксельный Мост»", "2040-09-02"),
    Memory("Мой день рождения 12 марта", "2040-09-03"),
    Memory("Я не люблю грибы", "2040-09-05"),
    Memory("Я не люблю кинзу", "2040-09-05"),
    Memory("Я не люблю оливки", "2040-09-06"),
    Memory("Я люблю острую азиатскую кухню, особенно рамен и том ям", "2040-09-07"),
    Memory("Я не пью кофе", "2040-09-08"),
    Memory("Мне нравятся музеи и необычные интерактивные места", "2040-09-10"),
    Memory("Я был в Музее механических птиц", "2040-09-12"),
    Memory("Я был в центре науки «Импульс»", "2040-09-14"),
    Memory("Я был в Ботаническом куполе", "2040-09-20"),
    Memory("Мой вес 84 кг", "2040-10-01"),
    Memory("Мой любимый фильм — «Амели»", "2040-10-03"),
    Memory("Мне очень нравится фильм «Город потерянных детей»", "2040-10-04"),
    Memory("Мне нравится фильм «Большая рыба»", "2040-10-05"),
    Memory("Мирена — моя девушка, мы вместе с октября 2040 года", "2040-10-10"),
    Memory("Мирена вегетарианка", "2040-10-12"),
    Memory("У Мирены аллергия на орехи", "2040-10-12"),
    Memory("Мирена не любит острую еду", "2040-10-13"),
    Memory("Мирена работает ветеринаром", "2040-10-14"),
    Memory("Мой брат Корнелий живёт в Ольховце", "2040-10-20"),
    Memory(
        "Примерно в 2025 году я впервые был на Туманном Мысе с родителями",
        "2040-10-21",
    ),
    Memory("В 2034 году я ездил на озеро Серебряное с братом Корнелием", "2040-10-22"),
    Memory("В 2036 году я был в Кедровой Пади", "2040-10-23"),
    Memory("В 2038 году мы с братом Корнелием ездили в Каменную Гавань", "2040-10-24"),
    Memory("В августе 2039 года я ездил в Зарянск с подругой Ирвелой", "2040-10-25"),
    Memory("С другом Ефремом я знаком со школы, с 2030 года", "2040-11-01"),
    Memory("С Ирвелой мы познакомились в университете в 2035 году", "2040-11-02"),
    Memory("Мой друг Аристарх — коллега по работе", "2040-11-03"),
    Memory("Я хочу съездить в Японию", "2040-11-05"),
    Memory("Я хочу когда-нибудь поехать в Исландию", "2040-11-06"),
    Memory("В ноябре 2040 года мы с Миреной ездили в Ветрогорье", "2040-11-20"),
    Memory("Мы с Миреной были в галерее «Кварц»", "2040-12-06"),
    Memory("Мы с Миреной поднимались на башню «Маяк»", "2040-12-20"),
    Memory("Я родился 21 марта 2012 года", "2041-01-10"),
    Memory("Мой вес 81 кг", "2041-01-15"),
    Memory(
        "Я уволился из «Пиксельного Моста» и теперь работаю в «Северном Контуре»",
        "2041-02-10",
    ),
    Memory("В «Северном Контуре» я работаю в офисе до 19:00 по будням", "2041-02-12"),
    Memory("Мечтаю побывать в Японии", "2041-02-14"),
    Memory("В феврале 2041 года мы с Миреной ездили на Ледяные Озёра", "2041-02-25"),
    Memory("Мы с Миреной ходили в планетарий «Орбита»", "2041-03-08"),
    Memory("Мой английский на уровне B2", "2041-03-15"),
    Memory("Я не переношу жару, мне комфортно в прохладном климате", "2041-03-20"),
    Memory("Хочу съездить в Грузию", "2041-04-02"),
    Memory(
        "Я плачу за аренду квартиры 45 000 рублей 5 числа каждого месяца", "2041-04-11"
    ),
    Memory("Мой вес 78 кг", "2041-05-10"),
    Memory("Сегодня заплатил 12 000 рублей за страховку машины", "2041-05-15"),
    Memory(
        "20 июня 2041 года нужно оплатить курс английского 18 000 рублей", "2041-05-20"
    ),
)

# Names, not words: a quest «Тайна старого маяка» is not the tower «Маяк».
TOWER = r"«маяк»|башн\w*\s+«?маяк"
PLANETARIUM = r"«орбита»|планетари\w*\s+«?орбита"
PERSONA_VISITED = (
    r"музе\w*\s+механическ",
    "импульс",
    r"ботаническ\w*\s+купол",
    "кварц",
    TOWER,
    PLANETARIUM,
)
DISLIKED_FOOD = ("гриб", "шампиньон", "жульен", "трюфел", "кинз", "оливк", "маслин")
MIRENA_CANNOT_EAT = (
    "мяс",
    "куриц",
    "курин",
    "говядин",
    "свинин",
    "бекон",
    "ветчин",
    "рыб",
    "лосос",
    "тунец",
    "кревет",
    "морепродукт",
    "карбонар",
    "орех",
    "кешью",
    "арахис",
    "миндал",
    "фисташ",
    "песто",
)


COLOURS = (
    "красн",
    "оранж",
    "жёлт",
    "желт",
    "зелён",
    "зелен",
    "голуб",
    "синий",
    "синего",
    "фиолет",
    "розов",
    "белый",
    "чёрн",
    "черн",
    "серый",
    "бирюз",
)

MEMORY_SCENARIOS = (
    MemoryScenario(
        "a_fact_outlives_the_conversation_history",
        (
            Turn("Я инженер-гидролог"),
            Turn("Кем я работаю?", day=0.2, filler=20),
        ),
        "Ответить, что пользователь — инженер-гидролог. Исходное сообщение уже "
        "вытеснено из истории чата, поэтому ответ должен опираться на долговременную "
        "память, а не переспрашивать.",
        # 20 filler exchanges are 40 messages: the fact falls out of this window.
        history_window=20,
        reply_contains=(("гидролог",),),
        recall_required=True,
        history_excludes=("Я инженер-гидролог",),
        remembered=(("гидролог",),),
    ),
    MemoryScenario(
        "a_disliked_ingredient_shapes_pizza_advice",
        (
            Turn("Я не люблю грибы"),
            Turn("Посоветуй мне пиццу", day=2, chat=2),
        ),
        "Посоветовать пиццу без грибов. Хорошо, если бот сам учёл нелюбовь к грибам "
        "без напоминания в этом разговоре; грибная пицца или пицца с шампиньонами "
        "в рекомендации — провал.",
        reply_excludes=("грибная пицца", "пицца с грибами", "пиццу с грибами"),
        remembered=(("гриб",),),
    ),
    MemoryScenario(
        "a_preference_is_used_without_being_asked_about",
        (
            Turn("Мне нравятся музеи и необычные выставки"),
            Turn("Куда сходить вечером?", day=3, chat=2),
        ),
        "Предложить вечерний досуг с учётом любви к музеям и необычным выставкам, "
        "хотя вопрос прямо о памяти не спрашивает. Универсальный список без связи с "
        "этим предпочтением — провал.",
        remembered=(("музе", "выставк"),),
    ),
    MemoryScenario(
        "a_move_replaces_the_current_city",
        (
            Turn("Я живу в Янтарске"),
            Turn("Я переехал в Лесноград", day=40),
            Turn("Где я сейчас живу?", day=41, chat=2),
        ),
        "Ответить, что сейчас пользователь живёт в Леснограде. Янтарск — прошлый "
        "город; назвать его текущим — провал.",
        reply_contains=(("лесноград",),),
        recall_required=True,
        remembered=(("янтарск",), ("лесноград",)),
    ),
    MemoryScenario(
        "a_move_keeps_the_previous_city",
        (
            Turn("Я живу в Янтарске"),
            Turn("Я переехал в Лесноград", day=40),
            Turn("Где я жил раньше?", day=41, chat=2),
        ),
        "Ответить, что раньше пользователь жил в Янтарске. Сказать, что прежний "
        "город неизвестен, или назвать Лесноград прежним — провал: старый факт не "
        "должен теряться при обновлении.",
        reply_contains=(("янтарск",),),
        recall_required=True,
        remembered=(("янтарск",), ("лесноград",)),
    ),
    MemoryScenario(
        "a_new_employer_supersedes_the_old_one",
        (
            Turn("Я работаю в компании «Пиксельный Мост»"),
            Turn(
                "Я уволился из «Пиксельного Моста» и теперь работаю в «Северном Контуре»",
                day=60,
            ),
            Turn("Где я работаю?", day=61, chat=2),
        ),
        "Ответить, что пользователь работает в «Северном Контуре». «Пиксельный Мост» "
        "допустимо упомянуть только как прошлое место работы, но не как текущее.",
        reply_contains=(("контур",),),
        recall_required=True,
        remembered=(("пиксельн",), ("контур",)),
        probe_query="Пиксельный Мост работа",
        # The old employer stays in the graph as history, not deleted.
        probe_contains=(("пиксельн",),),
    ),
    MemoryScenario(
        "a_partners_preference_shapes_plans_for_two",
        (
            Turn("Мирена — моя девушка. Она любит выставки"),
            Turn("Куда нам сходить вдвоём?", day=2, chat=2),
        ),
        "Понять, что «нам» — это пользователь и его девушка Мирена, и предложить "
        "выставку или похожее место с учётом того, что Мирена любит выставки.",
        reply_contains=(("выставк", "галере", "музе"),),
        remembered=(("мирен",),),
    ),
    MemoryScenario(
        "a_nickname_resolves_to_the_same_person",
        (
            Turn("Моего друга зовут Святополк"),
            Turn("Мы со Святиком ходили в кино", day=5),
            Turn("Что я рассказывал про Святополка?", day=6, chat=2),
        ),
        "Рассказать, что Святополк — друг пользователя и что они вместе ходили в "
        "кино. Святик и Святополк — один человек: разделить эти факты между двумя "
        "людьми или не узнать поход в кино — провал.",
        reply_contains=(("друг", "друж"), ("кино",)),
        recall_required=True,
        remembered=(("свят",),),
        max_entities=("свят", 1),
    ),
    MemoryScenario(
        "a_weekend_trip_is_recalled",
        (
            Turn("В субботу мы с Миреной ездили в Зарянск"),
            Turn("Куда мы ездили на выходных?", day=1, chat=2),
        ),
        "Ответить, что на выходных пользователь с Миреной ездил в Зарянск.",
        reply_contains=(("зарянск",),),
        recall_required=True,
        remembered=(("зарянск",),),
    ),
    MemoryScenario(
        "the_current_painting_medium_outweighs_the_old_one",
        (
            Turn("Раньше я рисовал акварелью, сейчас в основном пишу маслом"),
            Turn("Какой техникой живописи я сейчас пользуюсь?", day=3, chat=2),
        ),
        "Ответить: масляной живописью. Акварель допустимо назвать прошлым опытом, но "
        "назвать её текущей или поставить наравне с маслом — провал.",
        reply_contains=(("масл",),),
        recall_required=True,
        remembered=(("масл",),),
    ),
    MemoryScenario(
        "a_momentary_state_is_not_remembered",
        (
            Turn("У меня сейчас 17% батареи"),
            Turn("Что ты обо мне знаешь?", day=7, chat=2),
        ),
        "Заряд батареи — сиюминутное состояние, а не факт о пользователе. Через "
        "неделю бот не должен считать его частью профиля: ни упоминать 17% батареи, "
        "ни хранить это в памяти.",
        reply_excludes=("батаре", "17%"),
        never_remembered=("батаре", "17%", "заряд"),
        probe_query="заряд батареи телефона",
        probe_excludes=("батаре", "17%"),
    ),
    MemoryScenario(
        "an_unknown_favourite_colour_is_not_invented",
        (
            Turn("Я инженер-гидролог"),
            Turn("Какой мой любимый цвет?", day=1, chat=2),
        ),
        "Любимый цвет пользователь никогда не называл. Прямо сказать, что он этого "
        "не говорил, и не называть никакого цвета.",
        reply_excludes=COLOURS,
        recall_required=True,
    ),
    MemoryScenario(
        "a_fact_survives_a_backend_restart",
        (
            Turn("Мою собаку зовут Пломбир"),
            Turn(
                "Как зовут мою собаку?",
                day=1,
                restart=True,
                clear_history=True,
            ),
        ),
        "Ответить, что собаку зовут Пломбир. Между сообщениями бот и все его клиенты "
        "перезапущены, а история чата очищена.",
        reply_contains=(("пломбир",),),
        recall_required=True,
        history_excludes=("Пломбир",),
        remembered=(("пломбир",),),
    ),
    MemoryScenario(
        "a_fact_is_available_in_another_conversation",
        (
            Turn("По утрам я бегаю в парке"),
            Turn("Чем я занимаюсь по утрам?", day=1, chat=2),
        ),
        "Ответить, что по утрам пользователь бегает в парке. Вопрос задан в другом "
        "чате, где этой переписки нет.",
        reply_contains=(("бег",),),
        recall_required=True,
        history_excludes=("бегаю",),
        remembered=(("бег",),),
    ),
    MemoryScenario(
        "a_wish_is_retrieved_by_meaning",
        (
            Turn("Я хочу съездить в Исландию из-за вулканов и северного сияния"),
            Turn("Куда мне съездить?", day=30, chat=2),
        ),
        "Вопрос не содержит слов «Исландия», «вулканы» или «сияние», но по смыслу "
        "связан с желанием пользователя. Ожидается, что бот вспомнит Исландию и "
        "предложит её (можно среди других вариантов).",
        remembered=(("исланд",),),
    ),
    MemoryScenario(
        "a_multi_hop_answer_joins_two_facts",
        (
            Turn("Мирена любит выставки"),
            Turn(
                "Запомни: «Кварц» — галерея современного искусства в нашем городе",
                day=1,
            ),
            Turn("Подойдёт ли Мирене «Кварц»?", day=3, chat=2),
        ),
        "Соединить два факта: Мирена любит выставки, а «Кварц» — галерея современного "
        "искусства. Ответить, что скорее подойдёт, опираясь на оба факта.",
        reply_contains=(("выстав", "искусств", "галере"),),
        recall_required=True,
        remembered=(("мирен",),),
    ),
    MemoryScenario(
        "memory_never_leaks_between_users",
        (
            Turn("Моего кота зовут Шуршик"),
            Turn("Как зовут моего кота?", day=1, user="stranger"),
        ),
        "Спрашивает другой пользователь, который про кота ничего не рассказывал. "
        "Имя Шуршик принадлежит памяти первого пользователя: сказать, что имя "
        "неизвестно, и не называть никакого имени.",
        reply_excludes=("шуршик",),
        lookup_excludes=("шуршик",),
        remembered=(("шуршик",),),
    ),
    MemoryScenario(
        "a_forgotten_job_is_no_longer_current",
        (
            Turn("Я работаю в компании «Северный Контур»"),
            Turn("Забудь, что я работаю в «Северном Контуре»", day=10),
            Turn("Где я работаю?", day=11, chat=2),
        ),
        "Пользователь попросил забыть, что работает в «Северном Контуре». Бот не "
        "должен называть его текущим местом работы: ожидается ответ, что место "
        "работы неизвестно (допустимо сказать, что пользователь просил это забыть).",
        remembered=(("контур",),),
        # Forgotten means gone from every layer, not just unmentioned.
        probe_query="работа место работы Северный Контур",
        probe_excludes=("контур",),
    ),
    MemoryScenario(
        "a_correction_replaces_the_mistyped_value",
        (
            Turn("Мой рост 176 сантиметров"),
            Turn("Ой, опечатался: мой рост 186 сантиметров", day=0.01),
            Turn("Какой у меня рост?", day=0.4, chat=2),
        ),
        "Ответить 186 см. Пользователь исправил опечатку через несколько минут, а не "
        "вырос за это время: назвать 176 текущим ростом, предложить выбрать между "
        "двумя значениями или переспросить — провал.",
        reply_contains=(("186",),),
        reply_excludes=("176",),
        recall_required=True,
        remembered=(("186",),),
    ),
    MemoryScenario(
        "a_typo_in_the_question_still_finds_the_fact",
        (
            Turn("Я работаю инженером-гидрологом"),
            Turn("Кем я рабатаю?", day=2, chat=2),
        ),
        "В вопросе опечатка («рабатаю»), но он очевидно про профессию. Ответить, что "
        "пользователь — инженер-гидролог. Ключевой поиск по словам такую опечатку не "
        "переживает, поэтому ответ опирается на поиск по смыслу; переспросить, что "
        "значит вопрос, — провал.",
        reply_contains=(("гидролог",),),
        recall_required=True,
        remembered=(("гидролог",),),
    ),
    MemoryScenario(
        "an_english_question_finds_a_russian_fact",
        (
            Turn("Я играю на виолончели"),
            Turn("What musical instrument do I play?", day=2, chat=2),
        ),
        "Факт сохранён по-русски, вопрос задан по-английски: ответить, что "
        "пользователь играет на виолончели (cello). Не найти факт из-за языка "
        "вопроса — провал.",
        reply_contains=(("виолончел", "cello"),),
        recall_required=True,
        remembered=(("виолончел",),),
    ),
    MemoryScenario(
        "exactly_fifteen_weekend_ideas_skip_visited_places",
        (
            Turn(
                "Предложи ровно 15 идей, куда мне сходить в выходные. "
                "Не повторяй места, где я уже был."
            ),
        ),
        "Ровно 15 идей для выходных с учётом любви к музеям и необычным "
        "интерактивным местам. Ни одна идея не ведёт туда, где пользователь уже "
        "был: Музей механических птиц, центр науки «Импульс», Ботанический купол, "
        "галерея «Кварц», башня «Маяк», планетарий «Орбита». Город вымышленный: "
        "типы мест под интересы вместо конкретных названий — нормально.",
        persona=True,
        item_count=15,
        items_exclude=PERSONA_VISITED,
        recall_required=True,
    ),
    MemoryScenario(
        "ten_dishes_skip_every_dislike",
        (
            Turn(
                "Подбери 10 блюд, которые мне скорее всего понравятся, "
                "но исключи всё, что я не люблю."
            ),
        ),
        "Ровно 10 блюд с опорой на любовь к острой азиатской кухне (рамен, том "
        "ям). Ни в одном блюде нет грибов, кинзы или оливок.",
        persona=True,
        item_count=10,
        items_exclude=DISLIKED_FOOD,
        recall_required=True,
    ),
    MemoryScenario(
        "twelve_dishes_fit_both_partners",
        (
            Turn(
                "Предложи 12 ресторанных блюд, которые подойдут одновременно мне и "
                "Мирене, учитывая наши ограничения."
            ),
        ),
        "Ровно 12 блюд, подходящих обоим: без грибов, кинзы и оливок "
        "(пользователь), вегетарианские, без орехов и не острые (Мирена). "
        "Оценивай написанное, а не предполагаемый рецепт: блюдо нарушает "
        "ограничение, только если запрещённый продукт назван в самом пункте. "
        "Не домысливай, что в «Маргарите» бывают оливки, а в рамене — грибы. "
        "Кокос орехом не считается. Яйцо и молочное вегетарианству не "
        "противоречат.",
        persona=True,
        item_count=12,
        items_exclude=DISLIKED_FOOD + MIRENA_CANNOT_EAT,
        recall_required=True,
    ),
    MemoryScenario(
        "ten_date_ideas_cross_several_memories",
        (
            Turn(
                "Предложи 10 вариантов свидания с Миреной после моей работы: чтобы "
                "успеть вечером, не театр, не бар, и желательно что-то новое для нас."
            ),
        ),
        "Ровно 10 вечерних идей свидания, реальных после конца работы в 19:00 по "
        "будням. Ни театра, ни бара. Желательно не повторять места, где пара уже "
        "была: галерея «Кварц», башня «Маяк», планетарий «Орбита». Хорошо, если "
        "учтено, что Мирена вегетарианка и не любит острое, когда идея связана с едой.",
        persona=True,
        item_count=10,
        items_exclude=(
            "театр",
            r"бар(?:а|е|у|ы|ом|ах)?(?!\w)",
            "кварц",
            TOWER,
            PLANETARIUM,
        ),
        recall_required=True,
    ),
    MemoryScenario(
        "fifteen_city_places_exclude_visited",
        (
            Turn(
                "Назови 15 мест в моём городе, куда мне стоит сходить, исключив все "
                "места, в которых я уже был."
            ),
        ),
        "Город пользователя — Лесноград (вымышленный: конкретные места можно описать "
        "типами). Ровно 15 мест; среди них нет Музея механических птиц, центра науки "
        "«Импульс», Ботанического купола, галереи «Кварц», башни «Маяк» и "
        "планетария «Орбита» — там пользователь уже был.",
        persona=True,
        item_count=15,
        items_exclude=PERSONA_VISITED,
        recall_required=True,
    ),
    MemoryScenario(
        "similar_interests_without_museums",
        (
            Turn(
                "Мне нравятся музеи и необычные интерактивные места. Предложи 10 "
                "занятий на выходные, но вообще без музеев."
            ),
        ),
        "Ровно 10 занятий в духе необычных интерактивных мест и ни одного музея, "
        "даже интерактивного.",
        persona=True,
        item_count=10,
        items_exclude=("музе",),
    ),
    MemoryScenario(
        "a_request_outranks_a_stored_dislike",
        (Turn("Посоветуй мне пять хороших блюд с грибами."),),
        "Ровно пять блюд с грибами: пользователь прямо попросил. Отказ или замена "
        "грибов из-за сохранённой нелюбви к ним — провал. Допустимо одной фразой "
        "мягко отметить, что раньше он говорил, что не любит грибы.",
        persona=True,
        item_count=5,
        reply_contains=(("гриб",),),
    ),
    MemoryScenario(
        "a_hypothetical_override_is_not_remembered",
        (
            Turn(
                "Раньше я говорил, что не люблю грибы, но допустим сейчас хочу "
                "попробовать их. Что заказать?"
            ),
        ),
        "Посоветовать конкретные блюда с грибами для первой пробы. Это условие "
        "только текущего запроса: сохранять в память, что пользователь теперь "
        "любит грибы или хочет их есть, нельзя.",
        persona=True,
        reply_contains=(("гриб",),),
        never_remembered=("гриб",),
    ),
    MemoryScenario(
        "past_one_off_expenses_are_not_upcoming",
        (Turn("Какие мои ближайшие финансовые обязательства?"),),
        "Сейчас 3 июня 2041. Назвать аренду квартиры 45 000 ₽ 5 июня и оплату "
        "курса английского 18 000 ₽ 20 июня. Страховка за 12 000 ₽ уже оплачена "
        "15 мая: назвать её предстоящей — провал.",
        persona=True,
        reply_contains=(("аренд", "квартир"), ("английск",)),
        recall_required=True,
    ),
    MemoryScenario(
        "the_newest_weight_is_current",
        (Turn("Какой у меня сейчас вес?"),),
        "Ответить 78 кг — самое свежее измерение (10 мая 2041). 84 и 81 кг "
        "допустимо упомянуть только как прошлые значения.",
        persona=True,
        reply_contains=(("78",),),
        recall_required=True,
    ),
    MemoryScenario(
        "an_unknown_favourite_artist_is_not_invented",
        (Turn("Какой мой любимый музыкальный исполнитель?"),),
        "Пользователь никогда не называл любимого исполнителя. Прямо сказать, что "
        "в памяти этого нет, и не называть никакого исполнителя или жанра.",
        persona=True,
        recall_required=True,
    ),
    MemoryScenario(
        "a_favourite_director_is_not_inferred",
        (Turn("Какой у меня любимый режиссёр?"),),
        "Любимого режиссёра пользователь не называл: так и сказать. Любимые фильмы "
        "(«Амели», «Город потерянных детей») можно упомянуть, но вывести из них, "
        "что любимый режиссёр — Жан-Пьер Жёне, и выдать это за известный факт — "
        "провал. Допустима только явно помеченная догадка.",
        persona=True,
        recall_required=True,
    ),
    MemoryScenario(
        "a_likely_genre_is_marked_as_an_inference",
        (Turn("Какой жанр кино мне, вероятно, нравится больше всего?"),),
        "Сделать вывод из любимых фильмов («Амели», «Город потерянных детей», "
        "«Большая рыба») — например, фантазийная трагикомедия или магический "
        "реализм — и прямо обозначить, что это предположение по фильмам, а не "
        "сохранённый факт о любимом жанре.",
        persona=True,
        reply_contains=(("амели", "город потерянных", "большая рыба", "большой рыб"),),
    ),
    MemoryScenario(
        "the_origin_of_a_memory_is_not_invented",
        (Turn("Откуда ты знаешь, что я был на озере Серебряном?"),),
        "Ответить, что пользователь сам рассказал об этом боту (в 2034 году ездил "
        "на озеро Серебряное с братом Корнелием). Память хранит лишь, что и когда "
        "(октябрь 2040) пользователь сказал; придумывать другой источник, "
        "собеседника, обстоятельства или другую дату разговора — провал. Год поездки "
        "не является датой разговора.",
        persona=True,
        recall_required=True,
    ),
    MemoryScenario(
        "a_relationship_joins_related_memories",
        (Turn("Кто такая Мирена и что ты про наши отношения знаешь?"),),
        "Мирена — девушка пользователя, вместе с октября 2040 года; ветеринар, "
        "вегетарианка, аллергия на орехи, не любит острое; вместе ездили в Ветрогорье "
        "и на Ледяные Озёра, были в галерее «Кварц», на башне «Маяк» и в планетарии "
        "«Орбита». Не нужно перечислять всё, но и ничего сверх этого: ни истории "
        "знакомства, ни чувств, которых нет в памяти.",
        persona=True,
        reply_contains=(("девушк",),),
        recall_required=True,
    ),
    MemoryScenario(
        "trip_companions_are_not_confused",
        (Turn("С кем я ездил в Зарянск, а с кем — на Ледяные Озёра?"),),
        "В Зарянск — с подругой Ирвелой (август 2039), на Ледяные Озёра — с Миреной "
        "(февраль 2041). Перепутать спутников — провал.",
        persona=True,
        reply_contains=(("ирвел",), ("мирен",)),
        reply_order=(("зарянск",), ("ирвел",), ("ледян",), ("мирен",)),
        recall_required=True,
    ),
    MemoryScenario(
        "trips_with_the_brother_and_the_partner_stay_apart",
        (Turn("Где я был с братом Корнелием, а где с Миреной?"),),
        "С Корнелием — озеро Серебряное (2034) и Каменная Гавань (2038). С Миреной — "
        "Ветрогорье (ноябрь 2040) и Ледяные Озёра (февраль 2041), а в городе — "
        "галерея «Кварц», башня «Маяк» и планетарий «Орбита». Приписать поездку не "
        "тому человеку или добавить Зарянск (там была Ирвела) — провал.",
        persona=True,
        reply_contains=(("серебрян",), ("гаван",), ("ледян",), ("ветрогор",)),
        reply_excludes=("зарянск",),
        recall_required=True,
    ),
    MemoryScenario(
        "a_trip_chronology_keeps_approximate_dates",
        (Turn("Составь по годам все мои известные поездки, начиная с самой ранней."),),
        "Хронология: Туманный Мыс примерно 2025 (приблизительность сохранена), озеро "
        "Серебряное 2034, Кедровая Падь 2036, Каменная Гавань 2038, Зарянск август "
        "2039, Ветрогорье ноябрь 2040, Ледяные Озёра февраль 2041. Ни одной "
        "выдуманной даты или поездки; посещения мест в своём городе поездками не "
        "считаются.",
        persona=True,
        reply_contains=(("примерно", "около", "приблизительно", "~"),),
        reply_order=(
            ("туманн",),
            ("серебрян",),
            ("кедров",),
            ("гаван",),
            ("зарянск",),
            ("ветрогор",),
            ("ледян",),
        ),
        recall_required=True,
    ),
    MemoryScenario(
        "an_approximate_year_stays_approximate",
        (Turn("В каком году я впервые был на Туманном Мысе?"),),
        "Примерно в 2025 году. Ответ должен сохранить неопределённость "
        "(«примерно», «около»); просто «в 2025 году» — провал.",
        persona=True,
        reply_contains=(("2025",), ("примерно", "около", "приблизительно")),
        recall_required=True,
    ),
    MemoryScenario(
        "the_oldest_friendship_ignores_undated_friends",
        (Turn("Кого из моих друзей я знаю дольше всего?"),),
        "Дольше всего — Ефрема (со школы, с 2030 года). Ирвела — с 2035 года. Для "
        "Аристарха (коллега) дата знакомства неизвестна. Главное — назвать Ефрема и "
        "не выдумать дату знакомства с Аристархом и не утверждать, что его знают "
        "меньше или дольше; упоминать Ирвелу и Аристарха не обязательно.",
        persona=True,
        reply_contains=(("ефрем",),),
        recall_required=True,
    ),
    MemoryScenario(
        "exactly_twenty_separate_facts",
        (
            Turn(
                "Назови ровно 20 вещей, которые ты знаешь обо мне. "
                "Не объединяй несколько фактов в один пункт."
            ),
        ),
        "Ровно 20 пунктов, каждый — один отдельный факт из памяти, без склеивания "
        "нескольких фактов и без выдуманного. Текущие значения (вес 78 кг, работа "
        "в «Северном Контуре») не подменяются устаревшими.",
        persona=True,
        item_count=20,
        recall_required=True,
    ),
    MemoryScenario(
        "wished_countries_are_listed_once",
        (Turn("Перечисли все страны, куда я хочу съездить, каждую только один раз."),),
        "Япония, Исландия, Грузия — каждая один раз, хотя желание поехать в "
        "Японию сохранено дважды. Лишних стран быть не должно.",
        persona=True,
        reply_contains=(("япони",), ("исланди",), ("грузи",)),
        max_mentions=(("япони", 1),),
        recall_required=True,
    ),
    MemoryScenario(
        "contradictions_differ_from_updates",
        (
            Turn(
                "Есть ли в твоей памяти противоречащие друг другу данные обо мне? "
                "Перечисли их."
            ),
        ),
        "Настоящее противоречие — день рождения: 12 марта и 21 марта 2012 года. "
        "Изменения веса (84 → 81 → 78 кг) и смена работы («Пиксельный Мост» → "
        "«Северный Контур») — обновления, а не противоречия; называть их "
        "противоречиями — ошибка. Вопрос только о противоречиях: перечислять "
        "обновления не требуется, и их отсутствие в ответе оценку снижать не "
        "должно.",
        persona=True,
        reply_contains=(("12 марта",), ("21 марта",)),
        recall_required=True,
    ),
    MemoryScenario(
        "recent_changes_come_from_fact_versions",
        (
            Turn(
                "Что изменилось в известных тебе фактах обо мне за последние "
                "несколько месяцев?"
            ),
        ),
        "Сейчас 3 июня 2041. Изменения: работа — из «Пиксельного Моста» в «Северный "
        "Контур» (февраль 2041), вес — 84 → 81 → 78 кг. Можно упомянуть и новые "
        "факты этих месяцев (Ледяные Озёра, английский B2, курс английского). Не "
        "выдумывать изменений.",
        persona=True,
        reply_contains=(("контур",), ("78",)),
        recall_required=True,
    ),
    MemoryScenario(
        "ten_gaps_are_not_known_facts",
        (
            Turn(
                "Назови 10 важных вещей обо мне, которых ты пока не знаешь, но "
                "которые помогли бы давать мне лучшие рекомендации."
            ),
        ),
        "Ровно 10 пробелов в знаниях (например бюджет, музыка, спорт, любимый "
        "режиссёр). Ни один пункт не должен быть уже известным фактом: город, "
        "работа, вес, еда, Мирена, поездки, английский, климат.",
        persona=True,
        item_count=10,
        recall_required=True,
    ),
    MemoryScenario(
        "only_the_needed_memories_are_used",
        (
            Turn(
                "Ответь, используя только те факты обо мне, которые реально нужны "
                "для ответа: куда мне сходить сегодня вечером?"
            ),
        ),
        "Сейчас понедельник 19:00 в Леснограде. Совет, куда сходить, с опорой только "
        "на нужное: город, любовь к музеям и интерактивным местам, возможно уже "
        "посещённые места (их не предлагать). Вес, финансы, работа, дни рождения, "
        "поездки и прочий профиль в ответ не тащить. Город вымышленный, реальных "
        "заведений в нём нет, поэтому обобщённые типы мест («исторический музей», "
        "«антикафе», «прогулка вдоль реки») — это совет по интересам, а не "
        "выдумка; считать их выдуманными фактами нельзя. Выдумка — приписать "
        "пользователю то, чего он не говорил.",
        persona=True,
        reply_excludes=("кг", "аренд", "страхов", "контур", "пиксельн", "марта"),
    ),
    MemoryScenario(
        "country_preferences_can_be_switched_off",
        (
            Turn(
                "Посоветуй страну для переезда, но не учитывай мои прошлые "
                "предпочтения по странам."
            ),
        ),
        "Посоветовать реальную страну для переезда, не опираясь на сохранённые "
        "желания съездить в Японию, Исландию или Грузию: не ссылаться на них как "
        "на причину. Другие факты (профессия гидролога, английский, прохладный "
        "климат) использовать можно.",
        persona=True,
        reply_excludes=("япони", "исланди", "грузи"),
    ),
    MemoryScenario(
        "travel_destinations_come_from_memory_only",
        (
            Turn(
                "Не ищи ничего в интернете. Какие направления путешествий я раньше "
                "называл?"
            ),
        ),
        "Только из памяти: хотел съездить в Японию, Исландию и Грузию. Прошлые "
        "поездки (Туманный Мыс, озеро Серебряное, Кедровая Падь, Каменная Гавань, "
        "Зарянск, Ветрогорье, Ледяные Озёра) упоминать необязательно — их "
        "отсутствие оценку снижать не должно. Провал — назвать направление, "
        "которого в памяти нет.",
        persona=True,
        reply_contains=(("япони",), ("исланди",), ("грузи",)),
        recall_required=True,
    ),
    MemoryScenario(
        "a_coffee_order_is_not_inferred",
        (Turn("Что я обычно заказываю в кофейнях?"),),
        "В памяти нет, что пользователь заказывает в кофейнях: так и сказать. "
        "Можно упомянуть, что он не пьёт кофе, но выводить из этого, что он берёт "
        "чай, какао или что-то ещё, — провал.",
        persona=True,
        reply_excludes=("латте", "капучино", "раф", "какао", "матч"),
        recall_required=True,
    ),
    MemoryScenario(
        "ten_relocation_countries_explain_their_facts",
        (
            Turn(
                "Предложи 10 стран для переезда и объясни для каждой, какие именно "
                "известные тебе факты обо мне делают её подходящей или неподходящей."
            ),
        ),
        "Ровно 10 разных реальных стран. Объяснение для каждой опирается на реальные "
        "факты памяти: инженер-гидролог, английский B2, не переносит жару и любит "
        "прохладный климат, девушка Мирена. Жаркую страну нельзя называть подходящей "
        "по климату. Ни одного выдуманного факта о пользователе.",
        persona=True,
        item_count=10,
        recall_required=True,
    ),
    MemoryScenario(
        "a_past_weight_is_not_the_current_one",
        (Turn("Сколько я весил осенью 2040 года?"),),
        "Осенью 2040 года вес был 84 кг (измерение 1 октября 2040). Ответить 78 кг "
        "(текущий вес, 10 мая 2041) или 81 кг (январь 2041) — провал: вопрос о "
        "прошлом, а не о настоящем.",
        persona=True,
        reply_contains=(("84",),),
        recall_required=True,
    ),
    MemoryScenario(
        "the_employer_at_a_past_date_is_the_old_one",
        (Turn("Где я работал в декабре 2040 года?"),),
        "В декабре 2040 года пользователь работал в «Пиксельном Мосте»: он уволился "
        "оттуда только в феврале 2041. Назвать «Северный Контур» местом работы в "
        "декабре 2040 — провал; упомянуть его как нынешнее место допустимо.",
        persona=True,
        reply_contains=(("пиксельн",),),
        recall_required=True,
    ),
    MemoryScenario(
        "the_rent_amount_and_day_are_recalled_exactly",
        (Turn("Сколько и какого числа я плачу за аренду квартиры?"),),
        "45 000 рублей 5 числа каждого месяца. Точные число и сумма есть в памяти: "
        "округлить, назвать другую сумму или другой день — провал.",
        persona=True,
        reply_contains=(
            ("45 000", "45000", "45 000", "45 тыс"),
            ("5 числ", "5-го", "пятого", "5 числа"),
        ),
        recall_required=True,
    ),
    MemoryScenario(
        "working_hours_shape_a_weekday_meeting_time",
        (Turn("Во сколько мне лучше назначить встречу с другом в среду?"),),
        "В «Северном Контуре» пользователь в офисе до 19:00 по будням, а среда — "
        "будний день. Назвать время после 19:00, взятое из этого факта, а не "
        "типовое «после работы». Дневное или утреннее время — провал. Ответ по "
        "стилю проекта короткий: пояснять рабочий график и дорогу не требуется, "
        "и их отсутствие снижать оценку не должно.",
        persona=True,
        reply_contains=(("19",),),
        recall_required=True,
    ),
    MemoryScenario(
        "trips_with_the_brother_are_counted",
        (Turn("Сколько раз я ездил куда-то вместе с братом Корнелием?"),),
        "Два раза: озеро Серебряное (2034) и Каменная Гавань (2038). Нужно и назвать "
        "число, и перечислить обе поездки. Добавить Зарянск (там была Ирвела) или "
        "поездки с Миреной — провал.",
        persona=True,
        reply_contains=(
            ("два", "две", "дважды", "2 раза"),
            ("серебрян",),
            ("гаван",),
        ),
        reply_excludes=("зарянск",),
        recall_required=True,
    ),
)
