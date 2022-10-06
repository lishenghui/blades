from blades.clients.client import BladesClient


def test_base_client():
    id = "1"
    client = BladesClient(id=id)
    assert client.id() == id
