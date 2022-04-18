import attr


@attr.s(eq=False, frozen=False, slots=True)
class ValidationMessage:
    from_file = attr.ib(default=True,
                        validator=[attr.validators.instance_of(bool)])


@attr.s(eq=False, frozen=False, slots=True)
class CV_ValidationMessage(ValidationMessage):
    total_folds = attr.ib(default=5,
                          validator=[attr.validators.instance_of(int)])
    current_fold = attr.ib(default=0,
                           validator=[attr.validators.instance_of(int)])


@attr.s(eq=False, frozen=False, slots=True)
class HoldOut_ValidationMessage(ValidationMessage):
    val_split_percentage = attr.ib(default=0.0,
                                   validator=[attr.validators.instance_of(float)])


@attr.s(eq=False, frozen=False, slots=True)
class MetricMessage:
    loss = attr.ib()
    accuracy = attr.ib()
    precision = attr.ib()
    recall = attr.ib()
    f1 = attr.ib()
    confusion_matrix = attr.ib()


@attr.s(eq=False, frozen=False, slots=True)
class ServerMessage:
    new_model = attr.ib()
    validation_msg = attr.ib()
    send_deltas = attr.ib()
    target_label = attr.ib()


@attr.s(eq=False, frozen=False, slots=True)
class AckMessage:
    status = attr.ib(default=0)
    error = attr.ib(default='')


@attr.s(eq=False, frozen=False, slots=True)
class ClientMessage:
    new_model = attr.ib()
    train_metrics = attr.ib()
    validation_metrics = attr.ib()
    train_weight = attr.ib()
    validation_weight = attr.ib()
