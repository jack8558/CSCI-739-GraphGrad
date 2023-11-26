#pragma once

#include <iostream>
#include <unordered_map>
#include <vector>
#include <list>


// Hashmap that removes least recently used element if size exceeds.
template <typename Key, typename Value>
class LRUMap {
public:
    LRUMap(size_t initialCapacity) : maxCapacity(initialCapacity) {}

    void insert(const Key& key, const Value& value) {
        if (cacheMap.size() >= maxCapacity) {
            Key lruKey = lruList.back();
            lruList.pop_back();
            cacheMap.erase(lruKey);
        }

        cacheMap[key] = value;
        updateOrderOfAccess(key);
    }

   bool exists(const Key& key) {
        return cacheMap.find(key) != cacheMap.end();
    }

    std::optional<Value> get(const Key& key) {
        auto it = cacheMap.find(key);
        if (it != cacheMap.end()) {
            updateOrderOfAccess(key);
            return it->second;
        } else {
            return std::nullopt;  // Key not found
        }
    }

private:
    size_t maxCapacity;
    std::unordered_map<Key, Value> cacheMap;
    std::list<Key> lruList;

    void updateOrderOfAccess(const Key& key) {
        lruList.remove(key);
        lruList.push_front(key);
    }
};