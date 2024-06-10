from pydantic import BaseModel, SecretStr
from ..api import DBCaseConfig, DBConfig, MetricType


class OracleConfig(DBConfig):
    user: str
    password: SecretStr
    dsn: str

    def to_dict(self) -> dict:
        return {
            "user": self.user,
            "password": self.password.get_secret_value(),
            "dsn": self.dsn,
        }


class OracleDBCaseConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def _parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "L2_SQUARED"
        elif self.metric_type == MetricType.IP:
            return "DOT"
        return "COSINE"

    def index_param(self) -> dict:
        return {"distance": self._parse_metric()}

    def search_param(self) -> dict:
        return {"metric": self._parse_metric()}
