package main

import (
    "fmt"
    "math/rand"
    "time"
)

type Pulse struct {
    Hue   string
    Level int
}

func emitPulse() Pulse {
    hues := []string{"cyan", "magenta", "amber", "indigo"}
    rand.Seed(time.Now().UnixNano())
    return Pulse{
        Hue:   hues[rand.Intn(len(hues))],
        Level: rand.Intn(128),
    }
}

func main() {
    pulse := emitPulse()
    fmt.Printf("Aurora pulse => hue:%s level:%d\n", pulse.Hue, pulse.Level)
}



package main

import (
    "fmt"
    "math/rand"
    "time"
)

type Pulse struct {
    Hue   string
    Level int
}

func emitPulse() Pulse {
    hues := []string{"cyan", "magenta", "amber", "indigo"}
    rand.Seed(time.Now().UnixNano())
    return Pulse{
        Hue:   hues[rand.Intn(len(hues))],
        Level: rand.Intn(128),
    }
}

func main() {
    pulse := emitPulse()
    fmt.Printf("Aurora pulse => hue:%s level:%d\n", pulse.Hue, pulse.Level)
}


package main

import (
    "fmt"
    "math/rand"
    "time"
)

type Pulse struct {
    Hue   string
    Level int
}

func emitPulse() Pulse {
    hues := []string{"cyan", "magenta", "amber", "indigo"}
    rand.Seed(time.Now().UnixNano())
    return Pulse{
        Hue:   hues[rand.Intn(len(hues))],
        Level: rand.Intn(128),
    }
}

func main() {
    pulse := emitPulse()
    fmt.Printf("Aurora pulse => hue:%s level:%d\n", pulse.Hue, pulse.Level)
}
