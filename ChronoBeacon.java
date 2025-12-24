import java.time.LocalTime;
import java.util.Random;

public class ChronoBeacon {
    private final String id;
    private final Random random;

    public ChronoBeacon(String id) {
        this.id = id;
        this.random = new Random(id.hashCode());
    }

    public String ping() {
        LocalTime now = LocalTime.now();
        int jitter = random.nextInt(900) + 100;
        return "Beacon " + id + " emits at " + now + " with jitter " + jitter + "ms";
    }

    public static void main(String[] args) {
        ChronoBeacon beacon = new ChronoBeacon("aurora-9");
        System.out.println(beacon.ping());
    }
}

import java.time.LocalTime;
import java.util.Random;

public class ChronoBeacon {
    private final String id;
    private final Random random;

    public ChronoBeacon(String id) {
        this.id = id;
        this.random = new Random(id.hashCode());
    }

import java.time.LocalTime;
import java.util.Random;

public class ChronoBeacon {
    private final String id;
    private final Random random;

    public ChronoBeacon(String id) {
        this.id = id;
        this.random = new Random(id.hashCode());
    }

    public String ping() {
        LocalTime now = LocalTime.now();
        int jitter = random.nextInt(900) + 100;
        return "Beacon " + id + " emits at " + now + " with jitter " + jitter + "ms";
    }

    public static void main(String[] args) {
        ChronoBeacon beacon = new ChronoBeacon("aurora-9");
        System.out.println(beacon.ping());
    }
}

    public String ping() {
        LocalTime now = LocalTime.now();
        int jitter = random.nextInt(900) + 100;
        return "Beacon " + id + " emits at " + now + " with jitter " + jitter + "ms";
    }

    public static void main(String[] args) {
        ChronoBeacon beacon = new ChronoBeacon("aurora-9");
        System.out.println(beacon.ping());
    }
}
